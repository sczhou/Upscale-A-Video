import numpy as np
import os
import cv2
import torch
import torchvision

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".MP4", ".MOV", ".AVI")


def read_video_chunk(reader, chunk_size):
    frames = []
    try:
        for i in range(chunk_size):
            frame = reader.get_next_data()
            frame_np = np.array(frame)
            if frame_np.ndim == 2:  # Grayscale
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2RGB)
            if frame_np.shape[2] == 4:  # RGBA
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2RGB)
            frame_tensor = (
                torch.from_numpy(frame_np).permute(2, 0, 1).contiguous()
            )  # C, H, W
            frames.append(frame_tensor)
    except IndexError:
        pass
    except StopIteration:
        pass
    except Exception as e:
        print(f"Warning: Error reading frame during chunk read: {e}")
        pass
    if not frames:
        return None
    return torch.stack(frames)  # T C H W [0, 255]


def read_image_folder_chunk(folder_path, chunk_size, start_idx):
    files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(IMAGE_EXTENSIONS)]
    )
    total = len(files)
    if start_idx >= total:
        return None

    end_idx = min(start_idx + chunk_size, total)
    chunk_filenames = files[start_idx:end_idx]

    frames = []
    target_h, target_w = None, None
    for fname in chunk_filenames:
        img_path = os.path.join(folder_path, fname)
        frame_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if target_h is None:
            target_h, target_w = frame_rgb.shape[:2]
        if frame_rgb.shape[:2] != (target_h, target_w):
            frame_rgb = cv2.resize(
                frame_rgb, (target_w, target_h), interpolation=cv2.INTER_CUBIC
            )

        frame_tensor = (
            torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous()
        )  # C,H,W
        frames.append(frame_tensor)

    if not frames:
        return None

    return torch.stack(frames, dim=0)  # T,C,H,W


def read_frame_from_videos(frame_root):
    if frame_root.endswith(VIDEO_EXTENSIONS):  # Video file path
        video_name = os.path.basename(frame_root)[:-4]
        frames, _, info = torchvision.io.read_video(
            filename=frame_root, pts_unit="sec", output_format="TCHW"
        )  # RGB
        fps = info["video_fps"]
    else:
        video_name = os.path.basename(frame_root)
        frames = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            frame = cv2.imread(os.path.join(frame_root, fr))[..., [2, 1, 0]]  # RGB, HWC
            frames.append(frame)
        fps = None
        frames = torch.Tensor(np.array(frames)).permute(0, 3, 1, 2).contiguous()  # TCHW
    size = frames[0].size

    return frames, fps, size, video_name


def read_frames_chunk(is_video, video_reader, image_folder, chunk_size, current_idx=0):

    if is_video:
        frames = read_video_chunk(video_reader, chunk_size)
        return frames, None
    else:
        frames = read_image_folder_chunk(image_folder, chunk_size, current_idx)
        if frames is None:
            return None, None
        return frames, current_idx + (frames.shape[0])


def get_video_paths(input_root):
    video_paths = []
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(VIDEO_EXTENSIONS):
                video_paths.append(os.path.join(root, file))
    return sorted(video_paths)


def str_to_list(value):
    return list(map(int, value.split(",")))
