# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_condition.py

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

import math
import json

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import functional as F
from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.utils import BaseOutput, logging

try:
    from .unet_blocks import (
        CrossAttnDownBlock3D,
        CrossAttnUpBlock3D,
        DownBlock3D,
        UNetMidBlock3DCrossAttn,
        UpBlock3D,
        get_down_block,
        get_up_block,
    )
    from .resnet import InflatedConv3d
    from .temporal_module import TemporalModule3D, EmptyTemporalModule3D
except:
    from unet_blocks import (
        CrossAttnDownBlock3D,
        CrossAttnUpBlock3D,
        DownBlock3D,
        UNetMidBlock3DCrossAttn,
        UpBlock3D,
        get_down_block,
        get_up_block,
    )
    from resnet import InflatedConv3d
    from temporal_module import TemporalModule3D, EmptyTemporalModule3D
    
from rotary_embedding_torch import RotaryEmbedding

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads=8,
        num_buckets=32,
        max_distance=128,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype = torch.long, device = device)
        k_pos = torch.arange(n, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j') # num_heads, num_frames, num_frames

@dataclass
class UNet3DConditionOutput(BaseOutput):
    sample: torch.FloatTensor


class UNetVideoModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(
        self,
        ### Temporal Module Additional Kwargs ###
        down_temporal_idx = (0,1,2),
        mid_temporal = False,
        up_temporal_idx = (1,2,3),
        temporal_module_config = None,
        ###
        sample_size: Optional[int] = None, # 80
        in_channels: int = 7,
        out_channels: int = 4,
        center_input_sample: bool = False,
        max_noise_level: int = 350, 
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        block_out_channels: Tuple[int] = (
            256, 
            512, 
            512, 
            1024
        ),
        down_block_types: Tuple[str] = (
            "DownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D"
        ),
        mid_block_type: str = "UNetMidBlock3DCrossAttn",
        up_block_types: Tuple[str] = (
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "UpBlock3D"
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = (
            True,
            True,
            True,
            False
        ),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1024,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = True,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = 1000,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        use_first_frame: bool = False,
        use_relative_position: bool = False,
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = InflatedConv3d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        
        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim) # VSR for noise level
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # Temporal Modules
        self.down_temp_blocks = nn.ModuleList([])
        self.mid_temp_block = None
        self.up_temp_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        self.temporal_rotary_emb = RotaryEmbedding(32)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                use_first_frame=use_first_frame,
                use_relative_position=use_relative_position,
                rotary_emb=self.temporal_rotary_emb,
            )
            self.down_blocks.append(down_block)

            # Down Sample Temporal Modules
            down_temp_block = TemporalModule3D(
                in_channels=output_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                **temporal_module_config,
            ) if i in down_temporal_idx else EmptyTemporalModule3D()
            self.down_temp_blocks.append(down_temp_block)


        # mid
        if mid_block_type == "UNetMidBlock3DCrossAttn":
            self.mid_block = UNetMidBlock3DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
                use_first_frame=use_first_frame,
                use_relative_position=use_relative_position,
                rotary_emb=self.temporal_rotary_emb,
            )
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")

        self.mid_temp_block = TemporalModule3D(
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            **temporal_module_config,
        ) if mid_temporal else EmptyTemporalModule3D()

        # count how many layers upsample the videos
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                use_first_frame=use_first_frame,
                use_relative_position=use_relative_position,
                rotary_emb=self.temporal_rotary_emb,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

            up_temp_block = TemporalModule3D(
                in_channels=output_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                **temporal_module_config,
            ) if i in up_temporal_idx else EmptyTemporalModule3D()
            self.up_temp_blocks.append(up_temp_block)

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = InflatedConv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_slicable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_slicable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)

        num_slicable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_slicable_layers * [1]

        slice_size = num_slicable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock3D, DownBlock3D, CrossAttnUpBlock3D, UpBlock3D)):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        low_res: torch.FloatTensor,
        # encoder_hidden_states: torch.Tensor,
        encoder_hidden_states = None,
        class_labels: Optional[torch.Tensor] = 20,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ): # -> Union[UNet3DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, seq_length, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            class_labels: noise level
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None
        

        sample = torch.cat([sample, low_res], dim=1) # concat on C: 4+3=7
        
        #print(f'==============={sample.shape}================')
        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            # check noise level
            if torch.any(class_labels > self.config.max_noise_level):
                raise ValueError(f"`noise_level` has to be <= {self.config.max_noise_level} but is {class_labels}")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb


        # pre-process
        sample = self.conv_in(sample)

        # down
        down_block_res_samples = (sample,)
        for i,(downsample_block, down_temp_block) in enumerate(zip(self.down_blocks, self.down_temp_blocks)):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

            # 1. temporal modeling during down sample
            sample = down_temp_block(
                hidden_states=sample,
                encoder_hidden_states=encoder_hidden_states,
                timesteps=timesteps,
                temb=emb,
            )
            
                
        # mid
        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        )
        # 2. temporal modeling at mid block
        sample = self.mid_temp_block(
            hidden_states=sample,
            encoder_hidden_states=encoder_hidden_states,
            timesteps=timesteps,
            temb=emb,
        )
        # up
        for i, (upsample_block, up_temp_block) in enumerate(zip(self.up_blocks, self.up_temp_blocks)):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

            # 3. temporal modeling during up sample
            sample = up_temp_block(
                hidden_states=sample,
                encoder_hidden_states=encoder_hidden_states,
                timesteps=timesteps,
                temb=emb,
            )

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet3DConditionOutput(sample=sample)
    

    @classmethod
    def from_pretrained_2d(cls, config_path, pretrained_model_path):
        if not os.path.isfile(config_path):
            raise RuntimeError(f"{config_path} does not exist")
        with open(config_path, "r") as f:
            config = json.load(f)
        config["_class_name"] = cls.__name__

        model = cls.from_config(config)
        model_file = os.path.join(pretrained_model_path)
        if not os.path.isfile(model_file):
            raise RuntimeError(f"{model_file} does not exist")
        state_dict = torch.load(model_file, map_location="cpu")
        for k, v in model.state_dict().items():
            if 'temporal' in k: # newly added: 3DCNN + Temporal Attention
                print(f'New layers: {k}')
                state_dict.update({k: v})

        model.load_state_dict(state_dict, strict=True)

        for k, v in model.named_parameters():
            if not 'temporal' in k:
                v.requires_grad = False

        return model
    
if __name__ == '__main__':
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_path = "./configs/unet_3d_config.json"
    pretrained_model_path = "./pretrained_models/unet_diffusion_pytorch_model.bin"
    unet = UNet3DVSRModel.from_pretrained_2d(config_path, pretrained_model_path).to(device)
    
    noisy_latents = torch.randn((3, 4, 16, 32, 32)).to(device)
    bsz = noisy_latents.shape[0]
    timesteps = torch.randint(0, 1000, (bsz,)).to(device)
    timesteps = timesteps.long()
    encoder_hidden_states = torch.randn((bsz, 77, 768)).to(device)

    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    print(model_pred.shape)