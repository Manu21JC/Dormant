import os
import argparse
import torch
import numpy as np

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPVisionModelWithProjection
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
from einops import rearrange

from model_lib.models.unet_2d_condition import UNet2DConditionModel
from model_lib.models.unet_3d import UNet3DConditionModel
from model_lib.models.pose_guider import PoseGuider
from model_lib.models.appearance_encoder import AppearanceEncoderModel
from model_lib.ControlNet.ldm.util import instantiate_from_config
from utils.pgd import linfpgdattack
from utils.util import img2pose, seed_everything, enable_sequential_cpu_offload

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/protect.yaml")
    parser.add_argument("--ref_image_path", type=str, default="./inputs/000.png")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=30)

    parser.add_argument("--eps", type=int, default=16)
    parser.add_argument("--pgd_steps", type=int, default=200)
    parser.add_argument("--step_size", type=int, default=2)

    parser.add_argument("--output_dir", type=str, default="./outputs/")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    assert args.steps > 10, "The number of sampling steps should be greater than 10"

    config = OmegaConf.load(args.config)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    generator = seed_everything(args.seed)

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to(device, dtype=weight_dtype)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device=device)

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device)

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device=device
    )

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device=device)

    appearance_control_config = OmegaConf.load(config.appearance_control_config)
    appearance_control_model = instantiate_from_config(appearance_control_config.model).to(dtype=weight_dtype, device=device)

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)
    scheduler.set_timesteps(args.steps, device=device)

    width, height, length = args.W, args.H, args.L

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )

    appearance_encoder = AppearanceEncoderModel.from_pretrained(config.appearance_encoder_path, subfolder="appearance_encoder").to(dtype=weight_dtype, device=device)

    appearance_control_model.load_state_dict(
        torch.load(config.appearance_control_model_path, map_location="cpu"), strict=True
    )

    # memory efficient
    #if is_xformers_available():
    #    reference_unet.enable_xformers_memory_efficient_attention()
    #    denoising_unet.enable_xformers_memory_efficient_attention()
    #    appearance_encoder.enable_xformers_memory_efficient_attention() #####

    vae.enable_tiling()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    enable_sequential_cpu_offload(image_encoder, vae, reference_unet, pose_guider, device)

    pose_guider.requires_grad_(False)
    appearance_control_model.pose_control_model.requires_grad_(False)
    appearance_control_model.model.diffusion_model.requires_grad_(False)

    ref_name = Path(args.ref_image_path).stem
    ref_image_pil = Image.open(args.ref_image_path).convert("RGB").resize((width, height))
    ref_image_processor = VaeImageProcessor(
        vae_scale_factor=vae_scale_factor, do_convert_rgb=True
    )
    ref_image_tensor = ref_image_processor.preprocess(
        ref_image_pil, height=height, width=width
    )  # (bs, c, width, height) [-1, 1]
    ref_image_tensor = ref_image_tensor.to(
        dtype=weight_dtype, device=device
    )

    pose_image_pil = img2pose(ref_image_pil, device)
    pose_image_list = [pose_image_pil for _ in range(length)]
    cond_image_processor = VaeImageProcessor(
        vae_scale_factor=vae_scale_factor,
        do_convert_rgb=True,
        do_normalize=False,
    )
    pose_cond_tensor_list = []
    for pose_image in pose_image_list:
        pose_cond_tensor = cond_image_processor.preprocess(
            pose_image, height=height, width=width
        )
        pose_cond_tensor = pose_cond_tensor.unsqueeze(2)  # (bs, c, 1, h, w)
        pose_cond_tensor_list.append(pose_cond_tensor)
    pose_cond_tensor = torch.cat(pose_cond_tensor_list, dim=2)  # (bs, c, t, h, w)
    pose_cond_tensor = pose_cond_tensor.to(
        device=device, dtype=weight_dtype
    )

    attack = linfpgdattack(image_encoder, vae, denoising_unet, reference_unet, appearance_encoder, appearance_control_model, pose_guider, scheduler, generator, args)
    protect_image_tensor = attack.perturb(ref_image_tensor, pose_cond_tensor)

    protect_image_tensor = (protect_image_tensor.squeeze(0) / 2 + 0.5).clamp(0, 1)
    protect_image_array = 255. * rearrange(protect_image_tensor, 'c h w -> h w c').detach().cpu().numpy()
    protect_image_pil= Image.fromarray(protect_image_array.astype(np.uint8))

    output_name = ref_name + '_eps_' + str(args.eps) + '_steps_' + str(args.pgd_steps) + '_step_size_' + str(args.step_size) + '.png'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = os.path.join(args.output_dir, output_name)
    protect_image_pil.save(output_path)

if __name__ == "__main__":
    main()