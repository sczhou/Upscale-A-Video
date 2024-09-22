# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

import torch
import math
import imageio
import textwrap
import numpy as np
from PIL import Image
from einops import rearrange
from torch.nn import functional as F
from cog import BasePredictor, Input, Path
import transformers

transformers.logging.set_verbosity_error()

from models_video.RAFT.raft_bi import RAFT_bi
from models_video.propagation_module import Propagation
from models_video.autoencoder_kl_cond_video import AutoencoderKLVideo
from models_video.unet_video import UNetVideoModel
from models_video.pipeline_upscale_a_video import VideoUpscalePipeline
from models_video.scheduling_ddim import DDIMScheduler
from models_video.color_correction import (
    wavelet_reconstruction,
    adaptive_instance_normalization,
)
from llava.llava_agent import LLavaAgent
from utils import read_frame_from_videos, VIDEO_EXTENSIONS


MODEL_CACHE = "model_cache"
MODEL_URL = f"https://weights.replicate.delivery/default/sczhou/Upscale-A-Video/{MODEL_CACHE}.tar"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # load low_res_scheduler, text_encoder, tokenizer
        self.pipeline = VideoUpscalePipeline.from_pretrained(
            f"{MODEL_CACHE}/upscale_a_video", torch_dtype=torch.float16
        )

        # use_video_vae = False by default
        self.pipeline.vae = AutoencoderKLVideo.from_config(
            f"{MODEL_CACHE}/upscale_a_video/vae/vae_3d_config.json"
        )
        self.pipeline.vae.load_state_dict(
            torch.load(
                f"{MODEL_CACHE}/upscale_a_video/vae/vae_3d.bin", map_location="cpu"
            )
        )

        # load unet
        self.pipeline.unet = UNetVideoModel.from_config(
            f"{MODEL_CACHE}/upscale_a_video/unet/unet_video_config.json"
        )
        self.pipeline.unet.load_state_dict(
            torch.load(
                f"{MODEL_CACHE}/upscale_a_video/unet/unet_video.bin", map_location="cpu"
            ),
            strict=True,
        )
        self.pipeline.unet = self.pipeline.unet.half()
        self.pipeline.unet.eval()

        # load scheduler
        self.pipeline.scheduler = DDIMScheduler.from_config(
            f"{MODEL_CACHE}/upscale_a_video/scheduler/scheduler_config.json"
        )

        self.pipeline = self.pipeline.to("cuda:0")

        ## load LLaVA, model_id liuhaotian/llava-v1.5-13b
        self.llava_agent = LLavaAgent(
            f"{MODEL_CACHE}/liuhaotian-llava-v1.5-13b", load_8bit=True
        )

        self.raft = RAFT_bi(f"{MODEL_CACHE}/upscale_a_video/propagator/raft-things.pth")
        self.propagator = Propagation(4, learnable=False)

    def predict(
        self,
        input_video: Path = Input(description="Low-resolution video"),
        video_caption: str = Input(
            description="Optionally provide the video caption.", default=""
        ),
        use_llava: bool = Input(
            description="Use LLaVA to generate video caption. Ignored if video_caption is provided.",
            default=False,
        ),
        a_prompt: str = Input(
            description="Additional prompt", default="best quality, extremely detailed"
        ),
        n_prompt: str = Input(
            description="Negative prompt", default="blur, worst quality"
        ),
        noise_level: int = Input(
            description="Noise level applied to the input video. A higher noise level typically results in better video quality but lower fidelity.",
            default=150,
            le=200,
            ge=0,
        ),
        guidance_scale: float = Input(
            description="Classifier-free guidance scale for prompts. A higher guidance scale encourages the model to generate more details.",
            default=6,
            ge=0,
            le=20,
        ),
        inference_steps: int = Input(
            description="The number of denoising steps. More steps usually lead to a higher quality video.",
            default=30,
            ge=0,
            le=100,
        ),
        propagation_steps: str = Input(
            description="Optional. Propagation steps after performing denoising. Separate the steps woth comma.",
            default="",
        ),
        color_fix: str = Input(default="None", choices=["None", "AdaIn", "Wavelet"]),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        assert str(input_video).endswith(
            VIDEO_EXTENSIONS
        ), f"Video format needs to be one of {VIDEO_EXTENSIONS}"

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if len(propagation_steps.strip()) == 0:
            propagation_steps = []
            raft, propagator = None, None
        else:
            try:
                propagation_steps = list(map(int, propagation_steps.split(",")))
                raft, propagator = self.raft, self.propagator
            except ValueError as error:
                print(
                    f"Error: {error}. Ensure the input contains only integers separated by commas. Setting to empty propagation_steps."
                )
                propagation_steps = []
                raft, propagator = None, None

        self.pipeline.propagator = propagator

        llava_agent = self.llava_agent if use_llava else None

        ## ---------------------- start inferencing ----------------------
        vframes, fps, size, video_name = read_frame_from_videos(str(input_video))

        video_caption = video_caption.strip()
        if len(video_caption) > 0:
            use_llava = False

        if use_llava:
            print("Generating video caption with LLaVA...")
            with torch.no_grad():
                video_img0 = vframes[0]
                w, h = video_img0.shape[-1], video_img0.shape[-2]
                fix_resize = 512
                _upsacle = fix_resize / min(w, h)
                w *= _upsacle
                h *= _upsacle
                w0, h0 = round(w), round(h)
                video_img0 = F.interpolate(
                    video_img0.unsqueeze(0).float(), size=(h0, w0), mode="bicubic"
                )
                video_img0 = (
                    (video_img0.squeeze(0).permute(1, 2, 0))
                    .cpu()
                    .numpy()
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                video_img0 = Image.fromarray(video_img0)
                video_caption = llava_agent.gen_image_caption([video_img0])[0]

            wrapped_caption = textwrap.indent(
                textwrap.fill("Caption: " + video_caption, width=80), " " * 8
            )
            print(wrapped_caption)

        prompt = video_caption + " " + a_prompt
        print(f"Final prompt: {prompt}")

        vframes = (vframes / 255.0 - 0.5) * 2  # T C H W [-1, 1]
        vframes = vframes.to("cuda:0")

        h, w = vframes.shape[-2:]
        if h >= 1280 and w >= 1280:
            vframes = F.interpolate(vframes, (int(h // 4), int(w // 4)), mode="area")

        vframes = vframes.unsqueeze(dim=0)  # 1 T C H W
        vframes = rearrange(vframes, "b t c h w -> b c t h w").contiguous()  # 1 C T H W

        if raft is not None:
            flows_forward, flows_backward = raft.forward_slicing(vframes)
            flows_bi = [flows_forward, flows_backward]
        else:
            flows_bi = None

        b, c, t, h, w = vframes.shape
        generator = torch.Generator(device="cuda").manual_seed(seed)

        tile_size = 256
        perform_tile = False

        # For large resolution
        if h * w >= 384 * 384:
            perform_tile = True

        # ---------- Tile ----------
        torch.cuda.synchronize()
        if perform_tile:
            tile_height = tile_width = tile_size
            tile_overlap_height = tile_overlap_width = 64  # should be >= 64
            output_h = h * 4
            output_w = w * 4
            output_shape = (b, c, t, output_h, output_w)
            # start with black image
            output = vframes.new_zeros(output_shape)
            tiles_x = math.ceil(w / tile_width)
            tiles_y = math.ceil(h / tile_height)
            print(f"Processing the video w/ tile patches [{tiles_x}x{tiles_y}]...")

            rm_end_pad_w, rm_end_pad_h = True, True
            if (tiles_x - 1) * tile_width + tile_overlap_width >= w:
                tiles_x = tiles_x - 1
                rm_end_pad_w = False

            if (tiles_y - 1) * tile_height + tile_overlap_height >= h:
                tiles_y = tiles_y - 1
                rm_end_pad_h = False

            # loop over all tiles
            for y in range(tiles_y):
                for x in range(tiles_x):
                    print(f"\ttile: [{y+1}/{tiles_y}] x [{x+1}/{tiles_x}]")
                    # extract tile from input image
                    ofs_x = x * tile_width
                    ofs_y = y * tile_height
                    # input tile area on total image
                    input_start_x = ofs_x
                    input_end_x = min(ofs_x + tile_width, w)
                    input_start_y = ofs_y
                    input_end_y = min(ofs_y + tile_height, h)
                    # input tile area on total image with padding
                    input_start_x_pad = max(input_start_x - tile_overlap_width, 0)
                    input_end_x_pad = min(input_end_x + tile_overlap_width, w)
                    input_start_y_pad = max(input_start_y - tile_overlap_height, 0)
                    input_end_y_pad = min(input_end_y + tile_overlap_height, h)
                    # input tile dimensions
                    input_tile_width = input_end_x - input_start_x
                    input_tile_height = input_end_y - input_start_y
                    tile_idx = y * tiles_x + x + 1
                    input_tile = vframes[
                        :,
                        :,
                        :,
                        input_start_y_pad:input_end_y_pad,
                        input_start_x_pad:input_end_x_pad,
                    ]
                    if flows_bi is not None:
                        flows_bi_tile = [
                            flows_bi[0][
                                :,
                                :,
                                :,
                                input_start_y_pad:input_end_y_pad,
                                input_start_x_pad:input_end_x_pad,
                            ],
                            flows_bi[1][
                                :,
                                :,
                                :,
                                input_start_y_pad:input_end_y_pad,
                                input_start_x_pad:input_end_x_pad,
                            ],
                        ]
                    else:
                        flows_bi_tile = None

                    # upscale tile
                    try:
                        with torch.no_grad():
                            output_tile = self.pipeline(
                                prompt,
                                image=input_tile,
                                flows_bi=flows_bi_tile,
                                generator=generator,
                                num_inference_steps=inference_steps,
                                guidance_scale=guidance_scale,
                                noise_level=noise_level,
                                negative_prompt=n_prompt,
                                propagation_steps=propagation_steps,
                            ).images  # C T H W [-1, 1]
                    except RuntimeError as error:
                        print("Error", error)

                    # output tile area on total image
                    output_start_x = input_start_x * 4
                    if x == tiles_x - 1 and rm_end_pad_w == False:
                        output_end_x = output_w
                    else:
                        output_end_x = input_end_x * 4

                    output_start_y = input_start_y * 4
                    if y == tiles_y - 1 and rm_end_pad_h == False:
                        output_end_y = output_h
                    else:
                        output_end_y = input_end_y * 4

                    # output tile area without padding
                    output_start_x_tile = (input_start_x - input_start_x_pad) * 4
                    if x == tiles_x - 1 and rm_end_pad_w == False:
                        output_end_x_tile = (
                            output_start_x_tile + output_w - output_start_x
                        )
                    else:
                        output_end_x_tile = output_start_x_tile + input_tile_width * 4
                    output_start_y_tile = (input_start_y - input_start_y_pad) * 4
                    if y == tiles_y - 1 and rm_end_pad_h == False:
                        output_end_y_tile = (
                            output_start_y_tile + output_h - output_start_y
                        )
                    else:
                        output_end_y_tile = output_start_y_tile + input_tile_height * 4

                    # put tile into output image
                    output[
                        :,
                        :,
                        :,
                        output_start_y:output_end_y,
                        output_start_x:output_end_x,
                    ] = output_tile[
                        :,
                        :,
                        :,
                        output_start_y_tile:output_end_y_tile,
                        output_start_x_tile:output_end_x_tile,
                    ]
        else:
            print("Processing the video w/o tile...")
            try:
                with torch.no_grad():
                    output = self.pipeline(
                        prompt,
                        image=vframes,
                        flows_bi=flows_bi,
                        generator=generator,
                        num_inference_steps=inference_steps,
                        guidance_scale=guidance_scale,
                        noise_level=noise_level,
                        negative_prompt=n_prompt,
                        propagation_steps=propagation_steps,
                    ).images  # C T H W [-1, 1]
            except RuntimeError as error:
                print("Error", error)

        # color correction
        if color_fix in ["AdaIn", "Wavelet"]:
            vframes = rearrange(vframes.squeeze(0), "c t h w -> t c h w").contiguous()
            output = rearrange(output.squeeze(0), "c t h w -> t c h w").contiguous()
            vframes = F.interpolate(vframes, scale_factor=4, mode="bicubic")
            if color_fix == "AdaIn":
                output = adaptive_instance_normalization(output, vframes)
            elif color_fix == "Wavelet":
                output = wavelet_reconstruction(output, vframes)
        else:
            output = rearrange(output.squeeze(0), "c t h w -> t c h w").contiguous()

        output = output.cpu()

        torch.cuda.synchronize()

        upscaled_video = (output / 2 + 0.5).clamp(0, 1) * 255
        upscaled_video = rearrange(upscaled_video, "t c h w -> t h w c").contiguous()
        upscaled_video = upscaled_video.cpu().numpy().astype(np.uint8)

        out_path = "/tmp/out.mp4"
        imageio.mimwrite(
            out_path,
            upscaled_video,
            fps=fps,
            quality=8,
            output_params=["-loglevel", "error"],
        )  # Highest quality is 10, lowest is 0

        return Path(out_path)
