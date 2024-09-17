import os
import torch
import random
import pynvml
import gc
import numpy as np

from einops import rearrange
from PIL import Image
from pathlib import Path
from diffusers.utils import is_accelerate_available
from torchvision.transforms import v2 as T
from typing import Tuple, Union

from model_lib.dwpose import DWposeDetector

def seed_everything(seed):
    generator = torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)
    return generator

def enable_sequential_cpu_offload(image_encoder, vae, reference_unet, appearance_control_model, device):
    if is_accelerate_available():
        from accelerate import cpu_offload
    else:
        raise ImportError("Please install accelerate via `pip install accelerate`")

    for cpu_offloaded_model in [image_encoder, vae, reference_unet, appearance_control_model]:
        if cpu_offloaded_model is not None:
            cpu_offload(cpu_offloaded_model, device)

def get_memory_cost(gpu_id, period=None):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print("======{}======Memory cost: {}======".format(period, mem_info.used / float(1073741824)))
    pynvml.nvmlShutdown()

def img2pose(image_pil, device):
    detector = DWposeDetector()
    detector = detector.to(device)

    result, score = detector(image_pil)
    score = np.mean(score, axis=-1)

    del detector
    gc.collect()
    torch.cuda.empty_cache()

    return result

class NoTrans(torch.nn.Module):
    def forward(self, img_tensor):
        return img_tensor

class JPEG(torch.nn.Module):
    def __init__(self, quality: Union[int, Tuple[int, int]] = 75) -> None:
        super().__init__()
        self.quality = quality

    def forward(self, img_tensor):
        if isinstance(self.quality, tuple):
            quality = torch.randint(self.quality[0], self.quality[1], (1,)).item()
        else:
            quality = self.quality
        quality = torch.tensor([quality])
        device = img_tensor.device
        dtype = img_tensor.dtype
        from diff_jpeg import diff_jpeg_coding
        img_tensor = diff_jpeg_coding(img_tensor * 255, quality.to(device)).clamp(0, 255)
        img_tensor = img_tensor / 255.
        img_tensor = img_tensor.to(dtype).clamp(0, 1)
        return img_tensor

class GaussianNoise(torch.nn.Module):
    def __init__(self, mean: float = 0.0, sigma: Union[float, Tuple[float, float]] = 0.1, clip: bool = True) -> None:
        super().__init__()
        self.mean = mean
        self.sigma = sigma
        self.clip = clip

    def forward(self, img_tensor):
        if isinstance(self.sigma, tuple):
            sigma = torch.empty(1).uniform_(self.sigma[0], self.sigma[1]).item()
        else:
            sigma = self.sigma
        noise = self.mean + torch.randn_like(img_tensor) * sigma
        out = img_tensor + noise
        if self.clip:
            out = torch.clamp(out, 0, 1)
        return out

def transform(img_tensor):
    img_tensor = (img_tensor + 1) / 2
    width, height = img_tensor.shape[-1], img_tensor.shape[-2]
    trans = random.choice([T.GaussianBlur(kernel_size=3, sigma=(0.1, 3.0)), JPEG(quality=(30, 70)), GaussianNoise(sigma=(0, 0.03)), torch.nn.Sequential(T.RandomResize(min_size = int(height / 2), max_size = height * 2), T.Resize((height, width))), NoTrans()])
    img_tensor = trans(img_tensor).clamp(0, 1)
    img_tensor = (img_tensor * 2 - 1).clamp(-1, 1)
    return img_tensor

def decode_latents(vae, latents):
    video_length = latents.shape[2]
    latents = 1 / 0.18215 * latents
    latents = rearrange(latents, "b c f h w -> (b f) c h w")
    video = []
    with torch.no_grad():
        for frame_idx in range(latents.shape[0]):
            video.append(vae.decode(latents[frame_idx : frame_idx + 1]).sample)
    video = torch.cat(video)
    video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
    video = (video / 2 + 0.5).clamp(0, 1)
    video = video.detach().cpu().float().numpy()
    return video

def save_video(images, path="outputs/output.gif", fps=8):
    images = rearrange(images, "b c t h w -> t b c h w")
    outputs = []

    for x in images:
        x = np.squeeze(x * 255)
        x = x.transpose(1, 2, 0).astype(np.uint8)
        x = Image.fromarray(x)
        outputs.append(x)

    import av

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = outputs[0].size

    if save_fmt == ".mp4":
        codec = "libx264"
        container = av.open(path, "w")
        stream = container.add_stream(codec, rate=fps)

        stream.width = width
        stream.height = height

        for pil_image in outputs:
            av_frame = av.VideoFrame.from_image(pil_image)
            container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
        container.close()

    elif save_fmt == ".gif":
        outputs[0].save(
            fp=path,
            format="GIF",
            append_images=outputs[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")