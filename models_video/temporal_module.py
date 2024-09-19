from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import torchvision

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import FeedForward

try:
    from .diffusers_attention import CrossAttention
    from .resnet import Downsample3D, Upsample3D, InflatedConv3d, ResnetBlock3D, ResnetBlock3DCNN

except:
    from diffusers_attention import CrossAttention
    from resnet import Downsample3D, Upsample3D, InflatedConv3d, ResnetBlock3D, ResnetBlock3DCNN

from einops import rearrange, repeat
import math

import pdb


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def grid_sample_align(input, grid):
    return torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=True)


@dataclass
class TemporalTransformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class EmptyTemporalModule3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, w=1.0, encoder_hidden_states=None, timesteps=None, temb=None, attention_mask=None):
        return hidden_states


class TemporalModule3DVAE(nn.Module):
    def __init__(
        self,
        in_channels=None,
        s_resblock=False,
        double_t_resblock=False,
    ):
        super().__init__()
        
        self.s_resblock = s_resblock
        self.double_t_resblock = double_t_resblock
        self.non_linearity = nn.SiLU()
        self.resblocks_3d_temporal = ResnetBlock3DCNN(in_channels=in_channels, out_channels=in_channels, kernel=(3,1,1), temb_channels=None)
        if self.s_resblock:
            self.resblocks_3d_spatial = ResnetBlock3D(in_channels=in_channels, out_channels=in_channels, temb_channels=None, groups=32, groups_out=32)
        elif self.double_t_resblock: 
            self.resblocks_3d_spatial = zero_module(nn.Conv3d(in_channels, in_channels, kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0)))
        else:
            self.resblocks_3d_spatial = zero_module(InflatedConv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1))
        
        
        
    def forward(self, hidden_states, w=1.0):
        input_tensor = hidden_states
            
        # 3DCNN
        hidden_states = self.resblocks_3d_temporal(hidden_states)
        hidden_states = self.resblocks_3d_spatial(hidden_states)
        
        hidden_states = input_tensor + hidden_states * w

        return hidden_states



class TemporalModule3D(nn.Module):
    def __init__(
        self,
        in_channels=None,
        out_channels=None,

        num_attention_layers=1,
        num_attention_head=8,
        attention_head_dim=None,
        cross_attention_dim=768,
        temb_channels=512,

        dropout=0.,
        attention_bias=False,
        activation_fn="geglu",
        only_cross_attention=False,
        upcast_attention=False,

        norm_num_groups=8,
        use_linear_projection=True,
        use_scale_shift=False,

        attention_block_types: Tuple[str]=("",""),
        cross_frame_attention_mode=None,
        temporal_shift_fold_div=None,
        temporal_shift_direction="right",

        use_dcn_warpping=False,
        use_deformable_conv=True,

        attention_dim_div=2,
    ):
        super().__init__()
        assert len(attention_block_types) == 2

        self.use_scale_shift = use_scale_shift        
        self.non_linearity = nn.SiLU()
        
        # 1. 3d cnn
        self.resblocks_3d_temporal = ResnetBlock3DCNN(in_channels=in_channels, out_channels=in_channels, kernel=(5,1,1), temb_channels=temb_channels)
        self.resblocks_3d_spatial = ResnetBlock3D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, groups=32, groups_out=32)
        
        # 2. transformer blocks
        if not (attention_block_types[0]=='' and attention_block_types[1]==''):
            attentions = TemporalTransformer3DModel(
                num_attention_heads=num_attention_head,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else in_channels // num_attention_head // attention_dim_div,

                in_channels=in_channels,
                num_layers=num_attention_layers,
                dropout=dropout,
                norm_num_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attention_bias=attention_bias,
                activation_fn=activation_fn,
                num_embeds_ada_norm=1000, # adaptive norm for timestep embedding injection
                use_linear_projection=use_linear_projection,

                only_cross_attention=only_cross_attention,
                upcast_attention=upcast_attention,

                attention_block_types=attention_block_types,
                cross_frame_attention_mode=cross_frame_attention_mode,
                temporal_shift_fold_div=temporal_shift_fold_div,
                temporal_shift_direction=temporal_shift_direction,

                use_dcn_warpping=use_dcn_warpping,
                use_deformable_conv=use_deformable_conv,
            )
            self.attentions = nn.ModuleList([attentions])

        if use_scale_shift:
            self.scale_shift_conv = zero_module(InflatedConv3d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=1, stride=1, padding=0))
        else:
            self.shift_conv = zero_module(InflatedConv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0))


    def forward(self, hidden_states, w=1, encoder_hidden_states=None, timesteps=None, temb=None, attention_mask=None):
        input_tensor = hidden_states
            
        # 3DCNN
        hidden_states = self.resblocks_3d_temporal(hidden_states, temb)
        hidden_states = self.resblocks_3d_spatial(hidden_states, temb)
        
        if hasattr(self, "attentions"):
            for attn in self.attentions:
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, timestep=timesteps).sample

        if self.use_scale_shift:
            hidden_states = self.scale_shift_conv(hidden_states)
            scale, shift = torch.chunk(hidden_states, chunks=2, dim=1)
            hidden_states = (1 + scale) * input_tensor + shift
        else:
            hidden_states = self.shift_conv(hidden_states)
            hidden_states = input_tensor + hidden_states * w

        return hidden_states


class TemporalTransformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads=None,
        attention_head_dim=None,
        in_channels=None,
        num_layers=None,
        dropout=None,
        norm_num_groups=None,
        cross_attention_dim=None,
        attention_bias=None,
        activation_fn=None,
        num_embeds_ada_norm=None,
        use_linear_projection=None,
        only_cross_attention=None,
        upcast_attention=None,

        attention_block_types=None,
        cross_frame_attention_mode=None,
        temporal_shift_fold_div=2,
        temporal_shift_direction=None,

        use_dcn_warpping=None,
        use_deformable_conv=None,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,

                    attention_block_types=attention_block_types,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_shift_fold_div=temporal_shift_fold_div,
                    temporal_shift_direction=temporal_shift_direction,

                    use_dcn_warpping=use_dcn_warpping,
                    use_deformable_conv=use_deformable_conv,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(inner_dim, in_channels)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, return_dict: bool = True):
        # Input
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        if encoder_hidden_states is not None:
            encoder_hidden_states = repeat(encoder_hidden_states, 'b n c -> (b f) n c', f=video_length)

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length
            )

        # Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output,)

        return TemporalTransformer3DModelOutput(sample=output)


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim=None,
        num_attention_heads=None,
        attention_head_dim=None,
        dropout=None,
        cross_attention_dim=None,
        activation_fn=None,
        num_embeds_ada_norm=None,
        attention_bias=None,
        only_cross_attention=None,
        upcast_attention=None,

        attention_block_types=None,
        cross_frame_attention_mode=None,
        temporal_shift_fold_div=None,
        temporal_shift_direction=None,

        use_dcn_warpping=None,
        use_deformable_conv=None,
    ):
        super().__init__()
        assert len(attention_block_types) == 2

        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.use_dcn_warpping = use_dcn_warpping

        # 1. Spatial-Attn (self)
        if not attention_block_types[0] == '':
            self.attn_spatial = VersatileSelfAttention(
                attention_mode=attention_block_types[0],
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                cross_frame_attention_mode=cross_frame_attention_mode,
                temporal_shift_fold_div=temporal_shift_fold_div,
                temporal_shift_direction=temporal_shift_direction,
            )
            nn.init.zeros_(self.attn_spatial.to_out[0].weight.data)
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # 2. Temporal-Attn (self)
        self.attn_temporal = VersatileSelfAttention(
            attention_mode=attention_block_types[1],
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_shift_fold_div=temporal_shift_fold_div,
            temporal_shift_direction=temporal_shift_direction,
        )
        nn.init.zeros_(self.attn_temporal.to_out[0].weight.data)
        self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)


        self.dcn_module = WarpModule(
            in_channels=dim,
            use_deformable_conv=use_deformable_conv,
        ) if use_dcn_warpping else None

        # 3. Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)


    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool, attention_op: None):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            if hasattr(self, "attn_spatial"):
                self.attn_spatial._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None):
        # 1. Spatial-Attention
        if hasattr(self, "attn_spatial") and hasattr(self, "norm1"):
            norm_hidden_states = self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
            hidden_states = self.attn_spatial(norm_hidden_states, attention_mask=attention_mask, video_length=video_length) + hidden_states

        # 2. Temporal-Attention
        norm_hidden_states = self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
        if not self.use_dcn_warpping:
            hidden_states = self.attn_temporal(norm_hidden_states, attention_mask=attention_mask, video_length=video_length) + hidden_states
        else:
            hidden_states = self.dcn_module(
                hidden_states, 
                offset_hidden_states=self.attn_temporal(norm_hidden_states, attention_mask=attention_mask, video_length=video_length),
            )

        # 3. Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states


class VersatileSelfAttention(CrossAttention):
    def __init__(
            self,
            attention_mode=None,
            cross_frame_attention_mode=None,
            temporal_shift_fold_div=None,
            temporal_shift_direction=None,
            temporal_position_encoding=False,
            temporal_position_encoding_max_len=24,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        assert attention_mode in ("Temporal", "Spatial", "CrossFrame", "SpatialTemporalShift", None)
        assert cross_frame_attention_mode in ("0_i-1", "i-1_i", "0_i-1_i", "i-1_i_i+1", None)

        self.attention_mode = attention_mode

        self.cross_frame_attention_mode = cross_frame_attention_mode

        self.temporal_shift_fold_div = temporal_shift_fold_div
        self.temporal_shift_direction = temporal_shift_direction
        
        self.pos_encoder = PositionalEncoding(
            kwargs["query_dim"], 
            dropout=0., 
            max_len=temporal_position_encoding_max_len
        ) if temporal_position_encoding else None

    def temporal_token_concat(self, tensor, video_length):
        current_frame_index = torch.arange(video_length)
        former_frame_index = current_frame_index - 1
        former_frame_index[0] = 0
        
        later_frame_index = current_frame_index + 1
        later_frame_index[-1] = -1

        # (b f) d c
        tensor = rearrange(tensor, "(b f) d c -> b f d c", f=video_length)

        if self.cross_frame_attention_mode == "0_i-1":
            tensor = torch.cat([tensor[:, [0] * video_length], tensor[:, former_frame_index]], dim=2)
        elif self.cross_frame_attention_mode == "i-1_i":
            tensor = torch.cat([tensor[:, former_frame_index], tensor[:, current_frame_index]], dim=2)
        elif self.cross_frame_attention_mode == "0_i-1_i":
            tensor = torch.cat([tensor[:, [0] * video_length], tensor[:, former_frame_index], tensor[:, current_frame_index]], dim=2)
        elif self.cross_frame_attention_mode == "i-1_i_i+1":
            tensor = torch.cat([tensor[:, former_frame_index], tensor[:, current_frame_index], tensor[:, later_frame_index]], dim=2)
        elif self.cross_frame_attention_mode == None:
            tensor = tensor
        else:
            raise NotImplementedError        
        
        tensor = rearrange(tensor, "b f d c -> (b f) d c")
        return tensor

    def temporal_shift(self, tensor, video_length):
        # (b f) d c
        tensor = rearrange(tensor, "(b f) d c -> b f d c", f=video_length)
        n_channels = tensor.shape[-1]
        fold = n_channels // self.temporal_shift_fold_div

        if self.temporal_shift_direction != "right":
            raise NotImplementedError
        
        tensor_out = torch.zeros_like(tensor)
        tensor_out[:, 1:, :, :fold] = tensor[:, :-1, :, :fold]
        tensor_out[:, :, :, fold:]  = tensor[:, :, :, fold:]

        tensor_out = rearrange(tensor_out, "b f d c -> (b f) d c")
        return tensor_out

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        # pdb.set_trace()
        batch_size, sequence_length, _ = hidden_states.shape
        assert encoder_hidden_states is None

        # (b f) d c
        if self.attention_mode == "Temporal":
            d = hidden_states.shape[1]
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            
            if self.pos_encoder is not None:
                hidden_states = self.pos_encoder(hidden_states)

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        if self.attention_mode == "SpatialTemporalShift":
            key = self.temporal_shift(key, video_length=video_length)
            value = self.temporal_shift(value, video_length=video_length)
        elif self.attention_mode == "CrossFrame":
            key = self.temporal_token_concat(key, video_length=video_length)
            value = self.temporal_token_concat(value, video_length=video_length)

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)

        if self.attention_mode == "Temporal":
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states


class WarpModule(nn.Module):
    def __init__(
        self,
        in_channels=None,
        use_deformable_conv=None,
    ):
        super().__init__()
        self.use_deformable_conv = use_deformable_conv

        self.conv = None
        self.dcn_weight = None
        if use_deformable_conv:
            self.conv = nn.Conv2d(in_channels*2, 27, kernel_size=3, stride=1, padding=1)
            self.dcn_weight = nn.Parameter(torch.randn(in_channels, in_channels, 3, 3) / np.sqrt(in_channels * 3 * 3))
            self.alpha = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        else:
            self.conv = zero_module(nn.Conv2d(in_channels, 2, kernel_size=3, stride=1, padding=1))

    def forward(self, hidden_states, offset_hidden_states):
        # (b f) d c
        spatial_dim = hidden_states.shape[1]
        size = int(spatial_dim ** 0.5)
        assert size ** 2 == spatial_dim

        hidden_states = rearrange(hidden_states, "b (h w) c -> b c h w", h=size)
        offset_hidden_states = rearrange(offset_hidden_states, "b (h w) c -> b c h w", h=size)
        
        concat_hidden_states = torch.cat([hidden_states, offset_hidden_states], dim=1)

        input_tensor = hidden_states
        if self.use_deformable_conv:
            offset_x, offset_y, offsets_mask = torch.chunk(self.conv(concat_hidden_states), chunks=3, dim=1)
            offsets_mask = offsets_mask.sigmoid() * 2
            offsets = torch.cat([offset_x, offset_y], dim=1)
            hidden_states = torchvision.ops.deform_conv2d(
                hidden_states,
                offset=offsets,
                weight=self.dcn_weight,
                mask=offsets_mask,
                padding=1,
            )
            hidden_states = self.alpha * hidden_states + input_tensor

        else:
            offsets = self.conv(concat_hidden_states)
            hidden_states = self.optical_flow_warping(hidden_states, offsets)

        hidden_states = rearrange(hidden_states, "b c h w -> b (h w) c")
        return hidden_states

    @staticmethod
    def optical_flow_warping(x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        pad_mode (optional): ref to https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            "zeros": use 0 for out-of-bound grid locations,
            "border": use border values for out-of-bound grid locations
        """
        dtype = x.dtype
        if dtype != torch.float32:
            x = x.to(torch.float32)
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().to(flo.device)

        vgrid = grid + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = grid_sample_align(x, vgrid)

        mask = torch.ones_like(x)
        mask = grid_sample_align(x, vgrid)

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1
        results = output * mask
        if dtype != torch.float32:
            results = results.to(dtype)
        return results


class AdaLayerNorm(nn.Module):
    """
    Norm layer modified to incorporate timestep embeddings.
    """
    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x, timestep):
        # x shpae: (b f) n d / (b n) f d
        # timestep shape: b
        timestep = repeat(timestep, "b -> (b r)", r=x.shape[0] // timestep.shape[0])

        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1) # (b f) 1 2d
        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x

