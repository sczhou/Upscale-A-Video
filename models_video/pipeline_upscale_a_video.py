# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, List, Optional, Union
from tqdm import tqdm

import numpy as np
import math
import PIL
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.schedulers import DDPMScheduler
from diffusers.utils import deprecate, is_accelerate_available, is_accelerate_version, logging, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

from models_video.scheduling_ddim import DDIMScheduler
from models_video import AutoencoderKLVideo, UNetVideoModel, Propagation

from einops import rearrange

logger = logging.get_logger(__name__) 


def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = (x - x % 64 for x in (w, h))  # resize to integer multiple of 64

        image = [np.array(i.resize((w, h)))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


class VideoUpscalePipeline(DiffusionPipeline, TextualInversionLoaderMixin):
    def __init__(
        self,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        low_res_scheduler: DDPMScheduler,
        scheduler: DDIMScheduler = None,
        vae: AutoencoderKLVideo = None,
        unet: UNetVideoModel = None,
        propagator: Optional[Propagation] = None,
        max_noise_level: int = 350,
    ):
        super().__init__()

        if hasattr(
            vae, "config"
        ):  # check if vae has a config attribute `scaling_factor` and if it is set to 0.08333, else set it to 0.08333 and deprecate
            is_vae_scaling_factor_set_to_0_08333 = (
                hasattr(vae.config, "scaling_factor") and vae.config.scaling_factor == 0.08333
            )
            if not is_vae_scaling_factor_set_to_0_08333:
                deprecation_message = (
                    "The configuration file of the vae does not contain `scaling_factor` or it is set to"
                    f" {vae.config.scaling_factor}, which seems highly unlikely. If your checkpoint is a fine-tuned"
                    " version of `stabilityai/stable-diffusion-x4-upscaler` you should change 'scaling_factor' to"
                    " 0.08333 Please make sure to update the config accordingly, as not doing so might lead to"
                    " incorrect results in future versions. If you have downloaded this checkpoint from the Hugging"
                    " Face Hub, it would be very nice if you could open a Pull Request for the `vae/config.json` file"
                )
                deprecate("wrong scaling_factor", "1.0.0", deprecation_message, standard_warn=False)
                vae.register_to_config(scaling_factor=0.08333)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            low_res_scheduler=low_res_scheduler,
            scheduler=scheduler,
            propagator=propagator,
        )
        self.register_to_config(max_noise_level=max_noise_level)

    def to(self, device):
        self.vae = self.vae.to(device)
        self.text_encoder = self.text_encoder.to(device)
        self.unet = self.unet.to(device)
        if self.propagator is not None:
            self.propagator = self.propagator.to(device)
        
        return super(VideoUpscalePipeline, self).to(device)

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def decode_latents_vsr(self, latents, img, w_lr):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, img, w_lr).sample
        image = image.clamp(-1, 1).float()
        return image

    def check_inputs(
        self,
        prompt,
        image,
        noise_level,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or `list` but is {type(image)}"
            )

        # verify batch size of prompt and image are same if image is a list or tensor
        if isinstance(image, list) or isinstance(image, torch.Tensor):
            if isinstance(prompt, str):
                batch_size = 1
            else:
                batch_size = len(prompt)
            if isinstance(image, list):
                image_batch_size = len(image)
            else:
                image_batch_size = image.shape[0]
            if batch_size != image_batch_size:
                raise ValueError(
                    f"`prompt` has batch size {batch_size} and `image` has batch size {image_batch_size}."
                    " Please make sure that passed `prompt` matches the batch size of `image`."
                )

        # check noise level
        if noise_level > self.config.max_noise_level:
            raise ValueError(f"`noise_level` has to be <= {self.config.max_noise_level} but is {noise_level}")


    def prepare_latents_3d(self, batch_size, num_channels_latents, seq_len, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, seq_len, height, width)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]] = None,
        flows_bi: Optional[list] = None,
        num_inference_steps: int = 75,
        guidance_scale: float = 9.0,
        noise_level: int = 20,
        denoise_level: Optional[int] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        propagation_steps: list = [],
        w_lr: float = 1,
        return_dict: bool = True,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`PIL.Image.Image` or List[`PIL.Image.Image`] or `torch.FloatTensor`):
                `Image`, or tensor representing an image batch which will be upscaled. *
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. Ignored when not using guidance (i.e., ignored if `guidance_scale`
                is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
        """

        # 1. Check inputs
        self.check_inputs(
            prompt,
            image,
            noise_level,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        if image is None:
            raise ValueError("`image` input cannot be undefined.")

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Preprocess image
        # image = preprocess(image)
        image_dec = image.clone().to(dtype=torch.float32, device=device)
        image = image.to(dtype=prompt_embeds.dtype, device=device)

        # 5. Add noise to image
        noise_level = torch.tensor([noise_level], dtype=torch.long, device=device)
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
        image = self.low_res_scheduler.add_noise(image, noise, noise_level)

        batch_multiplier = 2 if do_classifier_free_guidance else 1
        image = torch.cat([image] * batch_multiplier * num_images_per_prompt)

        if denoise_level is None:
            denoise_level = torch.cat([noise_level] * image.shape[0])
        else:
            denoise_level = torch.tensor([denoise_level], dtype=torch.long, device=device)
            denoise_level = torch.cat([denoise_level] * image.shape[0])

        ####################### Random Noise ########################
        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # 6. Prepare latent variables        
        num_channels_latents = self.vae.config.latent_channels
        seq_len, height, width = image.shape[2:]
        latents = self.prepare_latents_3d(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            seq_len,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        ) # b c t h w

        # 7. Check that sizes of image and latents match
        num_channels_image = image.shape[1]
        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents+num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )

        # 8. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Denoising loop
        if not propagation_steps == []:
            print(" "*8 + "Propagation steps: " + str(propagation_steps))
        else:
            print(" "*8 + "Propagation steps: None")

        with tqdm(total=num_inference_steps, leave=True) as progress_bar:
            progress_bar.set_description(" "*8 + "Denoising")
            short_seq = 8
            overlap_seq = 2
                
            stride_seq = short_seq - overlap_seq
            vframes_seq = image.shape[2]

            for i, t in enumerate(timesteps):
                if flows_bi is not None and i in propagation_steps:
                    progress_bar.set_description(" "*8 + "Denoising w/ propagation")
                else:
                    progress_bar.set_description(" "*8 + "Denoising")
                torch.cuda.empty_cache() # clear for VSR
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if vframes_seq > short_seq:
                    noise_pred_list = [None]*vframes_seq
                    for start_f in range(0, vframes_seq, stride_seq):
                        torch.cuda.empty_cache()
                        end_f = min(vframes_seq, start_f + short_seq)
                        if end_f - start_f < short_seq: 
                            start_f = end_f - short_seq # keep f_num is 8 for each seq
                        noise_pred_ = self.unet(
                            latent_model_input[:,:,start_f:end_f], t, image[:,:,start_f:end_f], 
                            encoder_hidden_states=prompt_embeds, class_labels=denoise_level
                        ).sample
                        for k, idx in enumerate(list(range(start_f, end_f))):
                            if noise_pred_list[idx] is None:
                                noise_pred_list[idx] = noise_pred_[:,:,k:k+1]
                            else:
                                noise_pred_list[idx] = noise_pred_list[idx] * 0.5 + noise_pred_[:,:,k:k+1] * 0.5 # important
                    noise_pred = torch.cat(noise_pred_list, dim=2)
                else:
                    noise_pred = self.unet(
                        latent_model_input, t, image, encoder_hidden_states=prompt_embeds, class_labels=noise_level
                    ).sample

                
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # compute the previous noisy sample x_t -> x_t-1
                # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                x0 = self.scheduler.step_v0(noise_pred, t, latents, **extra_step_kwargs).pred_original_sample
                
                # performe feature propagation
                if flows_bi is not None and i in propagation_steps:
                    flows_forward, flows_backward = flows_bi[0].to(latents), flows_bi[1].to(latents)

                    x0 = self.propagator(x0, flows_forward, flows_backward, 
                                            interpolation='nearest', mode='fuse', fuse_scale=0.5,
                                            alpha1=0.001, alpha2=0.05)

                latents = self.scheduler.step_vt(x0, noise_pred, t, latents, **extra_step_kwargs).prev_sample

                progress_bar.update(1)
                        
                del latent_model_input, noise_pred
                

        # 10. Post-processing
        # make sure the VAE is in float32 mode, as it overflows in float16
        self.vae.to(dtype=torch.float32)
        latents = latents.float()

        # TODO(Patrick, William) - clean up when attention is refactored
        use_torch_2_0_attn = hasattr(F, "scaled_dot_product_attention")
        use_xformers = self.vae.decoder.mid_block.attentions[0]._use_memory_efficient_attention_xformers
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if not use_torch_2_0_attn and not use_xformers:
            self.vae.post_quant_conv.to(latents.dtype)
            self.vae.decoder.conv_in.to(latents.dtype)
            self.vae.decoder.mid_block.to(latents.dtype)
        else:
            latents = latents.float()

        latents_out = latents.clone()
        # 11. Convert to frames
        short_seq = 3
        # short_seq = 4
        video_length = latents.shape[2]
        if video_length > short_seq: # for VSR
            image = []
            start_list = list(range(0, latents.shape[2], short_seq))
            with tqdm(total=len(start_list), leave=True) as progress_bar:
                progress_bar.set_description(" "*8 + "Decoding")
                for i, start_f in enumerate(start_list):
                    torch.cuda.empty_cache() # delete for VSR
                    end_f = min(latents.shape[2], start_f + short_seq)
                    image_ = self.decode_latents_vsr(latents[:, :, start_f:end_f], image_dec[:, :, start_f:end_f], w_lr)
                    image.append(image_)
                    del image_
                    progress_bar.update(1)
            image = torch.cat(image, dim=2)
        else:
            image = self.decode_latents_vsr(latents, image_dec, w_lr)

        # performe image propagation
        # if flows_bi is not None:
        #     flows_forward, flows_backward = flows_bi[0], flows_bi[1]
        #     # image = self.propagator(image, flows_forward, flows_backward, interpolation='nearest')
        #     image = self.propagator(image, flows_forward, flows_backward, interpolation='bilinear', mode="fuse")

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, latents_out)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)