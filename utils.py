import os
import cv2
import torch
import torchvision

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI')

def read_frame_from_videos(frame_root):
    if frame_root.endswith(VIDEO_EXTENSIONS):  # Video file path
        video_name = os.path.basename(frame_root)[:-4]
        frames, _, info = torchvision.io.read_video(filename=frame_root, pts_unit='sec', output_format='TCHW') # RGB
        fps = info['video_fps']
    else:
        video_name = os.path.basename(frame_root)
        frames = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            frame = cv2.imread(os.path.join(frame_root, fr))[...,[2,1,0]] # RGB, HWC
            frames.append(frame)
        fps = None
        frames = torch.Tensor(np.array(frames)).permute(0, 3, 1, 2).contiguous() # TCHW
    size = frames[0].size

    return frames, fps, size, video_name


def get_video_paths(input_root):
    video_paths = []
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(VIDEO_EXTENSIONS):
                video_paths.append(os.path.join(root, file))
    return sorted(video_paths)

def str_to_list(value):
    return list(map(int, value.split(',')))