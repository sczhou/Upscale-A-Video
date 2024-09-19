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
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from diffusers.utils import BaseOutput, randn_tensor

try:
    from .unet_blocks import (
        UNetMidBlock3D,
        UNetMidBlock3D_plus,
        get_down_block,
        get_up_block,
    )
    from .resnet import InflatedConv3d, ResnetBlock3D, Fuse_sft_block, Fuse_CA_block, ResnetBlock3D_plus
    from .temporal_module import TemporalModule3DVAE, EmptyTemporalModule3D
except:
    from unet_blocks import (
        UNetMidBlock3D,
        UNetMidBlock3D_plus,
        get_down_block,
        get_up_block,
    )
    from resnet import InflatedConv3d, ResnetBlock3D, Fuse_sft_block, Fuse_CA_block, ResnetBlock3D_plus
    from temporal_module import TemporalModule3DVAE, EmptyTemporalModule3D
    
@dataclass
class DecoderOutput(BaseOutput):
    """
    Output of decoding method.

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Decoded output sample of the model. Output of the last layer of the model.
    """

    sample: torch.FloatTensor


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock3D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        double_z=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = InflatedConv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attn_num_head_channels=None,
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = InflatedConv3d(block_out_channels[-1], conv_out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x):
        sample = x
        sample = self.conv_in(sample)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            for down_block in self.down_blocks:
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(down_block), sample)

            # middle
            sample = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), sample)

        else:
            # down
            for down_block in self.down_blocks:
                sample = down_block(sample)

            # middle
            sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

# class Decoder(nn.Module):
#     def __init__(
#         self,
#         in_channels=3,
#         out_channels=3,
#         up_block_types=("UpDecoderBlock2D",),
#         block_out_channels=(64,),
#         layers_per_block=2,
#         norm_num_groups=32,
#         act_fn="silu",
#     ):
#         super().__init__()
#         self.layers_per_block = layers_per_block

#         self.conv_in = InflatedConv3d(
#             in_channels,
#             block_out_channels[-1],
#             kernel_size=3,
#             stride=1,
#             padding=1,
#         )

#         self.mid_block = None
#         self.up_blocks = nn.ModuleList([])

#         # mid
#         self.mid_block = UNetMidBlock3D(
#             in_channels=block_out_channels[-1],
#             resnet_eps=1e-6,
#             resnet_act_fn=act_fn,
#             output_scale_factor=1,
#             resnet_time_scale_shift="default",
#             attn_num_head_channels=None,
#             resnet_groups=norm_num_groups,
#             temb_channels=None,
#         )

#         # up
#         reversed_block_out_channels = list(reversed(block_out_channels))
#         output_channel = reversed_block_out_channels[0]
#         for i, up_block_type in enumerate(up_block_types):
#             prev_output_channel = output_channel
#             output_channel = reversed_block_out_channels[i]

#             is_final_block = i == len(block_out_channels) - 1

#             up_block = get_up_block(
#                 up_block_type,
#                 num_layers=self.layers_per_block + 1,
#                 in_channels=prev_output_channel,
#                 out_channels=output_channel,
#                 prev_output_channel=None,
#                 add_upsample=not is_final_block,
#                 resnet_eps=1e-6,
#                 resnet_act_fn=act_fn,
#                 resnet_groups=norm_num_groups,
#                 attn_num_head_channels=None,
#                 temb_channels=None,
#             )
#             self.up_blocks.append(up_block)
#             prev_output_channel = output_channel

#         # out
#         self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
#         self.conv_act = nn.SiLU()
#         self.conv_out = InflatedConv3d(block_out_channels[0], out_channels, 3, padding=1)

#         self.gradient_checkpointing = False

#     def forward(self, z):
#         sample = z
#         sample = self.conv_in(sample)

#         upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
#         if self.training and self.gradient_checkpointing:

#             def create_custom_forward(module):
#                 def custom_forward(*inputs):
#                     return module(*inputs)

#                 return custom_forward

#             # middle
#             sample = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), sample)
#             sample = sample.to(upscale_dtype)

#             # up
#             for up_block in self.up_blocks:
#                 sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample)
#         else:
#             # middle
#             sample = self.mid_block(sample)
#             sample = sample.to(upscale_dtype)

#             # up
#             for up_block in self.up_blocks:
#                 sample = up_block(sample)

#         # post-process
#         sample = self.conv_norm_out(sample)
#         sample = self.conv_act(sample)
#         sample = self.conv_out(sample)

#         return sample


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        up_block_types=("UpDecoderBlock3D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        condition_img=False,
        condition_channels=128,
        use_temporal_block=False
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.use_temporal_block = use_temporal_block
        self.condition_img = condition_img
        self.mid_block_type = "UNetMidBlock3D" if up_block_types[0] == "UpDecoderBlock3D" else "UNetMidBlock3D_plus"

        self.conv_in = InflatedConv3d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if self.condition_img:
            self.condition_in = nn.Sequential(
                ResnetBlock3D_plus(in_channels=3, out_channels=condition_channels, temb_channels=None, groups=3, groups_out=32),
                ResnetBlock3D_plus(in_channels=condition_channels, out_channels=condition_channels, temb_channels=None)
            )
            self.condition_fuse = Fuse_sft_block(condition_channels, block_out_channels[-1])

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        if self.use_temporal_block:
            self.mid_temporal_block = None
            self.up_temporal_blocks = nn.ModuleList([])

        # mid
        if self.mid_block_type == "UNetMidBlock3D":
            self.mid_block = UNetMidBlock3D(
                in_channels=block_out_channels[-1],
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                output_scale_factor=1,
                resnet_time_scale_shift="default",
                attn_num_head_channels=None,
                resnet_groups=norm_num_groups,
                temb_channels=None,
            )
        elif self.mid_block_type == "UNetMidBlock3D_plus":
            self.mid_block = UNetMidBlock3D_plus(
                in_channels=block_out_channels[-1],
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                output_scale_factor=1,
                resnet_time_scale_shift="default",
                attn_num_head_channels=None,
                resnet_groups=norm_num_groups,
                temb_channels=None,
            )
        else:
            raise ValueError(f"{self.mid_block_type} does not exist.")


        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = InflatedConv3d(block_out_channels[0], out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, z, img=None, w_lr=1.0):
        sample = z
        sample = self.conv_in(sample)

        # fuse input LQ image to decoder
        if self.condition_img:
            assert img is not None, "input img condition when condition_img is True."
            cond = self.condition_in(img)
            sample = self.condition_fuse(cond, sample, w=w_lr)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # middle
            sample = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), sample)
            sample = sample.to(upscale_dtype)

            # up
            for up_block in self.up_blocks:
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample)
        else:
            # middle
            sample = self.mid_block(sample)
            sample = sample.to(upscale_dtype)

            # up
            for up_block in self.up_blocks:
                sample = up_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape, generator=generator, device=self.parameters.device, dtype=self.parameters.dtype
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean
