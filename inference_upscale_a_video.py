# ================================================================ #
#   __  __                  __        ___     _   ___    __        #
#  / / / /__  ___ _______ _/ /__     / _ |   | | / (_)__/ /__ ___  #
# / /_/ / _ \(_-</ __/ _ `/ / -_) - / __ / - / |/ / / _  / -_) _ \ #
# \____/ .__/___/\__/\_,_/_/\__/   /_/ |_|   |___/_/\_,_/\__/\___/ #
#     /_/                                                          #
# ================================================================ #

import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
import logging

logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
import transformers

transformers.logging.set_verbosity_error()

import os
import cv2
import argparse
import sys

o_path = os.getcwd()
sys.path.append(o_path)

import torch
import torch.cuda
import time
import math
import json
import imageio
import textwrap
import pyfiglet
import numpy as np
import torchvision
from PIL import Image
from einops import rearrange
from torchvision.utils import flow_to_image, save_image
from torch.nn import functional as F

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
from utils import get_video_paths, str_to_list, read_frames_chunk
from utils import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from configs.CKPT_PTH import LLAVA_MODEL_PATH


if __name__ == "__main__":

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count >= 2:
            UAV_device = "cuda:0"
            LLaVA_device = "cuda:1"
            print(f"Detected {gpu_count} GPUs. Using {UAV_device} for Upscale-A-Video.")
        else:
            UAV_device = "cuda:0"
            LLaVA_device = "cuda:0"
            print(f"Detected 1 GPU. Using {UAV_device} for all models.")
    else:
        raise ValueError("Currently support CUDA only.")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_path", type=str, default="./inputs", help="Input folder."
    )
    parser.add_argument(
        "-o", "--output_path", type=str, default="./results", help="Output folder."
    )
    parser.add_argument(
        "-n",
        "--noise_level",
        type=int,
        default=120,
        help="Noise level [0, 200] applied to the input video. A higher noise level typically results in better \
                video quality but lower fidelity. Default value: 120",
    )
    parser.add_argument(
        "-g",
        "--guidance_scale",
        type=int,
        default=6,
        help="Classifier-free guidance scale for prompts. A higher guidance scale encourages the model to generate \
                more details. Default: 6",
    )
    parser.add_argument(
        "-s",
        "--inference_steps",
        type=int,
        default=30,  # 45 will add more details
        help="The number of denoising steps. More steps usually lead to a higher quality video. Default: 30",
    )
    parser.add_argument(
        "-p",
        "--propagation_steps",
        type=str_to_list,
        default=[],
        help="Propagation steps after performing denoising.",
    )
    parser.add_argument(
        "--a_prompt", type=str, default="best quality, extremely detailed"
    )
    parser.add_argument("--n_prompt", type=str, default="blur, worst quality")
    parser.add_argument("--use_video_vae", action="store_true", default=False)
    parser.add_argument(
        "--color_fix", type=str, default="None", choices=["None", "AdaIn", "Wavelet"]
    )
    parser.add_argument("--no_llava", action="store_true", default=False)
    parser.add_argument("--load_8bit_llava", action="store_true", default=False)
    parser.add_argument("--perform_tile", action="store_true", default=False)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--save_image", action="store_true", default=False)
    parser.add_argument("--save_suffix", type=str, default="")
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="Use FP16 precision."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4,
        help="Number of frames per chunk to process at once.",
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=32,
        help="Overlap (in pixels) for tiled processing.",
    )
    args = parser.parse_args()

    use_llava = not args.no_llava

    if args.fp16:
        torch_dtype = torch.float16
        print("Using FP16 precision for main pipeline on GPU.")
    else:
        torch_dtype = torch.float32
        print("Using FP32 precision for main pipeline on GPU.")

    print(pyfiglet.figlet_format("Upscale-A-Video", font="slant"))
    print(f"Upscale-A-Video Device: {UAV_device} ({torch_dtype})")
    if use_llava:
        print(f"LLaVA Device: {LLaVA_device} (8-bit: {args.load_8bit_llava})")
    else:
        print("LLaVA: Disabled")
    print(f"Processing in chunks of: {args.chunk_size} frames")
    print(
        f"Tiling: {'Enabled' if args.perform_tile else 'Automatic (if needed)'}, Tile Size: {args.tile_size}, Overlap: {args.tile_overlap}"
    )
    print(f"Color Correction: {args.color_fix}")
    print(
        f"Propagation Steps: {args.propagation_steps if args.propagation_steps else 'Disabled'}"
    )

    # load low_res_scheduler, text_encoder, tokenizer
    print("\nLoading Upscale-A-Video Pipeline components...")
    pipeline = VideoUpscalePipeline.from_pretrained(
        "./pretrained_models/upscale_a_video",
        torch_dtype=torch_dtype,
    )
    print("  - Pipeline structure loaded.")

    # load vae
    vae_config_path = "./pretrained_models/upscale_a_video/vae/"
    vae_model_path = "./pretrained_models/upscale_a_video/vae/"
    if args.use_video_vae:
        vae_config_path += "vae_video_config.json"
        vae_model_path += "vae_video.bin"
        print("  - Using Video VAE")
    else:
        vae_config_path += "vae_3d_config.json"
        vae_model_path += "vae_3d.bin"
        print("  - Using 3D VAE")
    pipeline.vae = AutoencoderKLVideo.from_config(vae_config_path)
    vae_state_dict = torch.load(vae_model_path, map_location="cpu")
    pipeline.vae.load_state_dict(vae_state_dict)
    print("  - VAE loaded to CPU.")

    # load unet
    unet_config_path = "./pretrained_models/upscale_a_video/unet/unet_video_config.json"
    unet_model_path = "./pretrained_models/upscale_a_video/unet/unet_video.bin"
    pipeline.unet = UNetVideoModel.from_config(unet_config_path)
    unet_state_dict = torch.load(unet_model_path, map_location="cpu")
    pipeline.unet.load_state_dict(unet_state_dict, strict=True)
    print("  - UNet loaded to CPU.")

    # load scheduler
    scheduler_config_path = (
        "./pretrained_models/upscale_a_video/scheduler/scheduler_config.json"
    )
    pipeline.scheduler = DDIMScheduler.from_config(scheduler_config_path)
    print("  - Scheduler loaded.")

    # load propagator
    raft = None
    propagator = None
    if args.propagation_steps:
        print("\nLoading RAFT for propagation...")

        raft = RAFT_bi(
            "./pretrained_models/upscale_a_video/propagator/raft-things.pth",
            device=UAV_device,
        )
        raft.eval()
        propagator = Propagation(4, learnable=False)
        propagator.eval()
    pipeline.propagator = propagator
    if propagator is not None:
        if torch_dtype == torch.float16:
            pipeline.propagator.half()
        elif torch_dtype == torch.float32:
            pipeline.propagator.float()
    print("RAFT model loaded. Propagator initialized.")

    print(f"  - Moving pipeline components to {UAV_device} with {torch_dtype}...")
    if torch_dtype == torch.float16:
        pipeline.vae.half()
        pipeline.unet.half()
    elif torch_dtype == torch.float32:
        pipeline.vae.float()
        pipeline.unet.float()
    pipeline.to(UAV_device)
    pipeline.vae.eval()
    pipeline.unet.eval()
    print("  - Pipeline components moved and configured.")

    ## load LLaVA
    video_captions = {}
    llava_agent = None
    if use_llava:
        print("\nLoading LLaVA...")
        llava_load_8bit = args.load_8bit_llava

        llava_agent = LLavaAgent(
            LLAVA_MODEL_PATH,
            device=LLaVA_device,
            load_8bit=llava_load_8bit,
            load_4bit=False,
        )
        print(f"LLaVA loaded on {LLaVA_device}.")

        if os.path.isfile(args.input_path) and args.input_path.lower().endswith(
            VIDEO_EXTENSIONS
        ):
            video_list_llava = [args.input_path]
        elif os.path.isdir(args.input_path):
            video_list_llava = get_video_paths(args.input_path)
        else:
            raise ValueError(f"Invalid input for LLaVA: '{args.input_path}'")

        if not video_list_llava:
            print(
                f"Warning: No video files found in {args.input_path} for LLaVA captioning."
            )
        else:
            print(f"Found {len(video_list_llava)} video(s) for captioning.")

        # gen captions
        for video_path in video_list_llava:
            print(
                f"  Generating caption for: {os.path.basename(video_path)} (using first frame)..."
            )
            reader_llava = imageio.get_reader(video_path)
            first_frame_raw = reader_llava.get_data(0)
            reader_llava.close()

            first_frame_np = np.array(first_frame_raw)
            if first_frame_np.ndim == 2:
                first_frame_np = cv2.cvtColor(first_frame_np, cv2.COLOR_GRAY2RGB)
            if first_frame_np.shape[2] == 4:
                first_frame_np = cv2.cvtColor(first_frame_np, cv2.COLOR_RGBA2RGB)
            h_llava_in, w_llava_in = (
                first_frame_np.shape[0],
                first_frame_np.shape[1],
            )

            with torch.no_grad():
                # resize for llava input using pillow
                fix_resize = 336
                _upscale = fix_resize / min(w_llava_in, h_llava_in)
                w_llava_proc, h_llava_proc = round(w_llava_in * _upscale), round(
                    h_llava_in * _upscale
                )
                video_img0_pil = Image.fromarray(first_frame_np).resize(
                    (w_llava_proc, h_llava_proc), Image.LANCZOS
                )

                caption = llava_agent.gen_image_caption([video_img0_pil])[0]
                video_captions[video_path] = caption
                wrapped_caption = textwrap.indent(
                    textwrap.fill("Caption: " + caption, width=80), " " * 6
                )
                print(wrapped_caption)

                if torch.cuda.is_available() and LLaVA_device != "cpu":
                    torch.cuda.synchronize(LLaVA_device)
                    torch.cuda.empty_cache()

    else:
        print("\nLLaVA captioning disabled.")

    ## input
    current_idx = 0

    def is_video_file(p):
        return p.lower().endswith(VIDEO_EXTENSIONS)

    def is_image_file(p):
        return p.lower().endswith(IMAGE_EXTENSIONS)

    if os.path.isfile(args.input_path):
        if is_video_file(args.input_path):
            is_video = True
            video_list = [args.input_path]
            image_folder = None
        elif is_image_file(args.input_path):
            is_video = False
            video_list = [os.path.dirname(args.input_path)]
            image_folder = os.path.dirname(args.input_path)
        else:
            raise ValueError("Unsupported file type given as input_path.")
    elif os.path.isdir(args.input_path):
        entries = sorted(
            f
            for f in os.listdir(args.input_path)
            if is_video_file(f) or is_image_file(f)
        )
        if not entries:
            raise ValueError("No images or videos found in the given directory.")

        first_entry = entries[0]
        if is_video_file(first_entry):
            is_video = True
            video_list = get_video_paths(args.input_path)
            image_folder = None
        else:
            is_video = False
            video_list = [args.input_path]
            image_folder = args.input_path
    else:
        raise ValueError("input_path must be a file or folder.")

    os.makedirs(args.output_path, exist_ok=True)

    print("Upscale-A-Video Pipeline ready.")

    ## ---------------------- start inferencing ----------------------
    total_videos = len(video_list)
    print(f"\nStarting processing for {total_videos} video(s)...\n" + "=" * 60)

    for i, video_path in enumerate(video_list):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        index_str = f"[{i+1}/{total_videos}]"
        print(f"{index_str} Processing video: {video_name}")
        print(f"  Input path: {video_path}")

        writer = None
        total_frames_processed = 0
        run_time = 0.0
        processing_successful = False
        if is_video:
            reader = imageio.get_reader(video_path)
            meta = reader.get_meta_data()
            fps = max(meta.get("fps", 30), 1)

            n_frames = reader.count_frames()
            print(f"  Detected ~{n_frames} frames.")

            first_frame_raw = next(reader.iter_data())
            first_frame_np = np.array(first_frame_raw)
            if first_frame_np.ndim == 2:
                first_frame_np = cv2.cvtColor(first_frame_np, cv2.COLOR_GRAY2RGB)
            if first_frame_np.shape[2] == 4:
                first_frame_np = cv2.cvtColor(first_frame_np, cv2.COLOR_RGBA2RGB)
            h, w = first_frame_np.shape[:2]
            print(f"  Video resolution: {w}x{h}, FPS: {fps:.2f}")
        else:
            current_idx = 0
            reader = None
            fps = 30
            n_frames = len(
                [
                    f
                    for f in os.listdir(video_path)
                    if f.lower().endswith(IMAGE_EXTENSIONS)
                ]
            )
            print(f"  Detected {n_frames} images.")
            first_image_path = sorted(os.listdir(video_path))[0]
            frame_bgr = cv2.imread(os.path.join(video_path, first_image_path))
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            print(f"  Image folder resolution: {w}x{h}, FPS: {fps}")

        video_caption = video_captions.get(video_path, "")
        if use_llava:
            if video_caption:
                print(
                    f"  Using generated caption: {textwrap.shorten(video_caption, width=70)}"
                )
            else:
                print(f"  No caption generated or available for this video.")
        prompt = video_caption + " " + args.a_prompt

        prop_suffix = (
            "_p" + "_".join(map(str, args.propagation_steps))
            if args.propagation_steps
            else ""
        )
        tile_suffix = (
            f"_t{args.tile_size}o{args.tile_overlap}"
            if args.perform_tile or (h * w >= 384 * 384)
            else ""
        )
        cf_suffix = f"_cf-{args.color_fix}" if args.color_fix != "None" else ""
        prec_suffix = "_fp32" if not args.fp16 else ""
        user_suffix = "_" + args.save_suffix if args.save_suffix else ""
        save_name = f"{video_name}_n{args.noise_level}_g{args.guidance_scale}_s{args.inference_steps}{prop_suffix}{tile_suffix}{cf_suffix}{prec_suffix}{user_suffix}"

        save_video_root = os.path.join(args.output_path, "video")
        os.makedirs(save_video_root, exist_ok=True)
        save_video_path = os.path.join(save_video_root, f"{save_name}.mp4")
        output_h = h * 4
        output_w = w * 4
        print(f"  Output path: {save_video_path}")
        writer = imageio.get_writer(
            save_video_path,
            fps=fps,
            codec="libx264",
            quality=None,
            bitrate=None,
            output_params=[
                "-crf",
                "20",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
                "-loglevel",
                "error",
            ],
        )

        # chunking
        prev_tail_frame = None
        generator = torch.Generator(device=UAV_device).manual_seed(10)
        chunk_idx = 0
        start_time = time.time()
        if is_video and reader is not None:
            reader.close()
            reader = imageio.get_reader(video_path)
        else:
            pass

        while True:
            print(f"\n{index_str} Reading chunk {chunk_idx + 1}...")
            vframes_chunk_cpu, next_idx = read_frames_chunk(
                is_video=is_video,
                video_reader=reader,
                image_folder=image_folder,
                chunk_size=args.chunk_size,
                current_idx=current_idx,
            )  # T C H W [0, 255], CPU

            if vframes_chunk_cpu is None:
                if is_video:
                    print(f"{index_str} End of video reached.")
                else:
                    print(f"{index_str} End of image folder reached.")
                break

            if prev_tail_frame is not None:
                vframes_chunk_cpu = torch.cat(
                    [prev_tail_frame, vframes_chunk_cpu], dim=0
                )

            if not is_video:
                current_idx = next_idx

            current_chunk_size = vframes_chunk_cpu.shape[0]
            print(
                f"{index_str} Processing chunk {chunk_idx + 1} ({current_chunk_size} frames)..."
            )

            # preprocess chunk: T C H W [0, 255] -> B C T H W [-1, 1]
            vframes_chunk_processed = (vframes_chunk_cpu / 255.0 - 0.5) * 2.0
            vframes_chunk_processed = vframes_chunk_processed.unsqueeze(0).permute(
                0, 2, 1, 3, 4
            )  # 1 C T H W
            vframes_chunk = vframes_chunk_processed.to(
                device=UAV_device, dtype=torch_dtype
            )

            b, c, t_chunk, h_chunk, w_chunk = vframes_chunk.shape

            # calculate optical flow
            flows_bi_chunk = None
            if raft is not None and t_chunk > 1:
                print(f"\tCalculating flow for chunk {chunk_idx + 1}...")
                with torch.no_grad():
                    raft_input = vframes_chunk.float()
                    flows_forward, flows_backward = raft.forward_slicing(raft_input)

                    flows_bi_chunk = [
                        flows_forward,
                        flows_backward,
                    ]
                print(f"\tFlow calculated.")
            else:
                flows_bi_chunk = None

            perform_tile_chunk = args.perform_tile or (h * w >= 384 * 384)
            output_chunk = None

            with torch.no_grad():
                if perform_tile_chunk:
                    tile_height = min(args.tile_size, h)
                    tile_width = min(args.tile_size, w)
                    tile_overlap = min(args.tile_overlap, tile_height // 2, tile_width // 2)

                    # seam blend
                    scale = output_h // h
                    output_shape_chunk = (b, c, t_chunk, output_h, output_w)
                    accum_chunk = torch.zeros(output_shape_chunk, dtype=torch_dtype, device=UAV_device)
                    weight_chunk = torch.zeros((1, 1, 1, output_h, output_w), dtype=torch_dtype, device=UAV_device)

                    tiles_x = math.ceil(w / tile_width)
                    tiles_y = math.ceil(h / tile_height)
                    print(f"\tProcessing chunk w/ tile patches [{tiles_x}x{tiles_y}], size={tile_width}x{tile_height}, overlap={tile_overlap}...")

                    for y in range(tiles_y):
                        for x in range(tiles_x):
                            input_start_y = y * tile_height
                            input_end_y = min(input_start_y + tile_height, h)
                            input_start_x = x * tile_width
                            input_end_x = min(input_start_x + tile_width, w)

                            pad_y_start = max(0, input_start_y - tile_overlap)
                            pad_y_end   = min(h, input_end_y + tile_overlap)
                            pad_x_start = max(0, input_start_x - tile_overlap)
                            pad_x_end   = min(w, input_end_x + tile_overlap)

                            input_tile = vframes_chunk[:, :, :, pad_y_start:pad_y_end, pad_x_start:pad_x_end]

                            flows_bi_tile = None
                            if flows_bi_chunk is not None:
                                if (flows_bi_chunk[0] is not None and flows_bi_chunk[0].ndim == 5
                                    and flows_bi_chunk[1] is not None and flows_bi_chunk[1].ndim == 5):
                                    flows_f_tile = flows_bi_chunk[0][:, :, :, pad_y_start:pad_y_end, pad_x_start:pad_x_end]
                                    flows_b_tile = flows_bi_chunk[1][:, :, :, pad_y_start:pad_y_end, pad_x_start:pad_x_end]
                                    if flows_f_tile.numel() > 0 and flows_b_tile.numel() > 0:
                                        flows_bi_tile = [flows_f_tile, flows_b_tile]

                            output_tile = pipeline(
                                prompt=prompt,
                                image=input_tile,
                                flows_bi=flows_bi_tile,
                                generator=generator,
                                num_inference_steps=args.inference_steps,
                                guidance_scale=args.guidance_scale,
                                noise_level=args.noise_level,
                                negative_prompt=args.n_prompt,
                                propagation_steps=(args.propagation_steps if (raft is not None and t_chunk > 1) else []),
                            ).images  # B C T H_pad_up W_pad_up

                            out_pad_y0 = pad_y_start * scale
                            out_pad_x0 = pad_x_start * scale
                            gen_h, gen_w = output_tile.shape[-2:]
                            out_pad_y1 = min(out_pad_y0 + gen_h, output_h)
                            out_pad_x1 = min(out_pad_x0 + gen_w, output_w)
                            place_h = out_pad_y1 - out_pad_y0
                            place_w = out_pad_x1 - out_pad_x0
                            if place_h <= 0 or place_w <= 0:
                                continue

                            output_tile = output_tile[:, :, :, :place_h, :place_w]

                            # effective overlaps
                            eff_top    = (input_start_y - pad_y_start) * scale
                            eff_bottom = (pad_y_end - input_end_y) * scale
                            eff_left   = (input_start_x - pad_x_start) * scale
                            eff_right  = (pad_x_end - input_end_x) * scale

                            fade_top    = int(min(max(eff_top,    0), place_h))
                            fade_bottom = int(min(max(eff_bottom, 0), place_h))
                            fade_left   = int(min(max(eff_left,   0), place_w))
                            fade_right  = int(min(max(eff_right,  0), place_w))

                            # build separable cosine hann feather mask
                            y_idx = torch.arange(place_h, device=UAV_device, dtype=output_tile.dtype)
                            wy_top = (0.5 - 0.5 * torch.cos(math.pi * torch.clamp(y_idx / fade_top, 0, 1))) if fade_top > 0 else torch.ones_like(y_idx)
                            wy_bot = (0.5 - 0.5 * torch.cos(math.pi * torch.clamp((place_h - 1 - y_idx) / fade_bottom, 0, 1))) if fade_bottom > 0 else torch.ones_like(y_idx)
                            wy = torch.minimum(wy_top, wy_bot)

                            x_idx = torch.arange(place_w, device=UAV_device, dtype=output_tile.dtype)
                            wx_left  = (0.5 - 0.5 * torch.cos(math.pi * torch.clamp(x_idx / fade_left, 0, 1))) if fade_left > 0 else torch.ones_like(x_idx)
                            wx_right = (0.5 - 0.5 * torch.cos(math.pi * torch.clamp((place_w - 1 - x_idx) / fade_right, 0, 1))) if fade_right > 0 else torch.ones_like(x_idx)
                            wx = torch.minimum(wx_left, wx_right)

                            mask2d = (wy.view(place_h, 1) * wx.view(1, place_w)).clamp_(0, 1)
                            mask = mask2d.view(1, 1, 1, place_h, place_w)  # broadcast over B,C,T

                            accum_chunk[:, :, :, out_pad_y0:out_pad_y1, out_pad_x0:out_pad_x1] += output_tile * mask
                            weight_chunk[:, :, :, out_pad_y0:out_pad_y1, out_pad_x0:out_pad_x1] += mask

                    eps = torch.finfo(accum_chunk.dtype).eps
                    output_chunk = accum_chunk / weight_chunk.clamp_min(eps)

                else:
                    print(f"\tProcessing chunk w/o tile...")
                    output_chunk = pipeline(
                        prompt=prompt,
                        image=vframes_chunk,
                        flows_bi=flows_bi_chunk,
                        generator=generator,
                        num_inference_steps=args.inference_steps,
                        guidance_scale=args.guidance_scale,
                        noise_level=args.noise_level,
                        negative_prompt=args.n_prompt,
                        propagation_steps=(args.propagation_steps if (raft is not None and t_chunk > 1) else []),
                    ).images  # B C T H_up W_up [-1, 1]

                # post-process chunk
                # output is B C T H W [-1, 1] on UAV_device
                output_chunk = rearrange(
                    output_chunk.squeeze(0), "c t h w -> t c h w"
                ).contiguous()

                # color correction
                if args.color_fix in ["AdaIn", "Wavelet"] and t_chunk > 1:
                    print(f"\tApplying {args.color_fix} color correction...")
                    with torch.no_grad():
                        vframes_chunk_orig_float = (
                            vframes_chunk_cpu.to(device=UAV_device, dtype=torch.float32)
                            / 255.0
                        )  # T C H W [0,1] float32 for resizing
                        vframes_chunk_resized = F.interpolate(
                            vframes_chunk_orig_float,
                            size=(output_h, output_w),
                            mode="bicubic",
                            align_corners=False,
                        )

                        output_chunk_corrected = output_chunk

                        if args.color_fix == "AdaIn":
                            # needs [0, 1]
                            output_chunk_01 = (output_chunk / 2.0 + 0.5).clamp(0, 1)
                            output_chunk_corrected_01 = adaptive_instance_normalization(
                                output_chunk_01, vframes_chunk_resized
                            )
                            # back to [-1, 1]
                            output_chunk_corrected = (
                                output_chunk_corrected_01 * 2.0 - 1.0
                            )
                        elif args.color_fix == "Wavelet":
                            # needs [0, 1]
                            output_chunk_01 = (output_chunk / 2.0 + 0.5).clamp(0, 1)
                            output_chunk_corrected_01 = wavelet_reconstruction(
                                output_chunk_01, vframes_chunk_resized
                            )
                            # back to [-1, 1]
                            output_chunk_corrected = (
                                output_chunk_corrected_01 * 2.0 - 1.0
                            )

                        output_chunk = output_chunk_corrected

                    print(f"\tColor correction applied.")

                output_chunk_cpu = output_chunk.float().cpu()

                # T C H W [-1, 1] -> T H W C [0, 255] uint8
                output_chunk_cpu = (output_chunk_cpu / 2.0 + 0.5).clamp(0, 1) * 255.0
                output_chunk_cpu = rearrange(
                    output_chunk_cpu, "t c h w -> t h w c"
                ).contiguous()
                output_chunk_uint8 = output_chunk_cpu.numpy().astype(np.uint8)

                print(f"\tWriting {output_chunk_uint8.shape[0]} frames to video...")
                start_write_idx = (
                    1 if (prev_tail_frame is not None and chunk_idx > 0) else 0
                )
                for frame_idx in range(start_write_idx, output_chunk_uint8.shape[0]):
                    # for frame_idx in range(output_chunk_uint8.shape[0]):
                    writer.append_data(output_chunk_uint8[frame_idx])
                    if args.save_image:
                        frame_num = total_frames_processed + (
                            frame_idx - start_write_idx
                        )
                        if frame_idx == 0 and chunk_idx == 0:
                            save_img_root = os.path.join(args.output_path, "frame")
                            save_img_dir = os.path.join(save_img_root, save_name)
                            os.makedirs(save_img_dir, exist_ok=True)
                            print(f"\tSaving individual frames to: {save_img_dir}")

                        save_img_path = os.path.join(
                            save_img_dir, f"{str(frame_num).zfill(6)}.png"
                        )
                        imageio.imwrite(save_img_path, output_chunk_uint8[frame_idx])

                processed_count_in_chunk = output_chunk_uint8.shape[0]
                total_frames_processed += processed_count_in_chunk - start_write_idx
                print(
                    f"{index_str} Finished processing chunk {chunk_idx + 1}. Total frames processed: {total_frames_processed}"
                )
                processing_successful = True

            prev_tail_frame = vframes_chunk_cpu[-1:].clone()
            chunk_idx += 1

        end_time = time.time()
        run_time = end_time - start_time
        writer.close()
        prev_tail_frame = prev_tail_flow_f = prev_tail_flow_b = None
        if reader is not None:
            reader.close()

    print(f"\nAll video processing finished. Results are in {args.output_path}")
