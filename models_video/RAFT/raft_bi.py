import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from einops import rearrange

# from models_video.RAFT import RAFT
from .raft import RAFT

def resize_flow_pytorch(flow, newh, neww):
    oldh, oldw = flow.shape[-2:]
    flow = F.interpolate(flow, (newh, neww), mode='bilinear')
    flow[:, :, 0] *= newh / oldh
    flow[:, :, 1] *= neww / oldw
    return flow


def initialize_RAFT(model_path='pretrained_models/raft-things.pth', device='cuda'):
    """Initializes the RAFT model.
    """
    args = argparse.ArgumentParser()
    args.raft_model = model_path
    args.small = False
    args.mixed_precision = False
    args.alternate_corr = False
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.raft_model, map_location='cpu'))
    model = model.module

    model.to(device)

    return model

class RAFT_bi(nn.Module):
    """Flow completion loss"""
    def __init__(self, model_path='weights/raft-things.pth', device='cuda'):
        super().__init__()
        self.fix_raft = initialize_RAFT(model_path, device=device)

        for p in self.fix_raft.parameters():
            p.requires_grad = False

        self.l1_criterion = nn.L1Loss()
        self.eval()

    def forward(self, gt_local_frames, iters=20):
        B, C, T, H, W = gt_local_frames.size()
        divisor = 8
        H_ = int(ceil(H/divisor) * divisor)
        W_ = int(ceil(W/divisor) * divisor)

        gt_local_frames = F.interpolate(gt_local_frames, (T, H_, W_), mode='trilinear')

        with torch.no_grad():
            gtlf_1 = rearrange(gt_local_frames[:, :, :-1, :, :], 'b c t h w -> (b t) c h w').contiguous()
            gtlf_2 = rearrange(gt_local_frames[:, :, 1:, :, :], 'b c t h w -> (b t) c h w').contiguous()

            _, gt_flows_forward = self.fix_raft(gtlf_1, gtlf_2, iters=iters, test_mode=True)
            _, gt_flows_backward = self.fix_raft(gtlf_2, gtlf_1, iters=iters, test_mode=True)

        gt_flows_forward = resize_flow_pytorch(gt_flows_forward, H, W)
        gt_flows_backward = resize_flow_pytorch(gt_flows_backward, H, W)

        gt_flows_forward = rearrange(gt_flows_forward, '(b t) c h w -> b c t h w', t=T-1).contiguous()
        gt_flows_backward = rearrange(gt_flows_backward, '(b t) c h w -> b c t h w', t=T-1).contiguous()

        return gt_flows_forward, gt_flows_backward
    

    def forward_slicing(self, gt_local_frames, iters=20):
        # ---- compute flow ----
        if gt_local_frames.size(-1) <= 640: 
            short_clip_len = 12
        elif gt_local_frames.size(-1) <= 720: 
            short_clip_len = 8
        elif gt_local_frames.size(-1) <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2

        # use fp32 for RAFT
        video_length = gt_local_frames.size(2)
        if video_length > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in range(0, video_length, short_clip_len):
                end_f = min(video_length, f + short_clip_len)
                if f == 0:
                    flows_f, flows_b = self.forward(gt_local_frames[:, :, f:end_f], iters=iters)
                else:
                    flows_f, flows_b = self.forward(gt_local_frames[:, :, f-1:end_f], iters=iters)
                
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                torch.cuda.empty_cache()
                
            gt_flows_f = torch.cat(gt_flows_f_list, dim=2)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=2)
        else:
            gt_flows_f, gt_flows_b = self.forward(gt_local_frames, iters=iters)
            torch.cuda.empty_cache()
        
        # print('gt_flows_f', gt_flows_f.shape)
        return gt_flows_f, gt_flows_b
