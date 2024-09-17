
<h2 align="center">Dormant: Defending against Pose-driven Human Image Animation</h2>

<p align="center">
  <video autoplay controls muted loop height="100%">
    <source src="./assets/teaser.mp4"
            type="video/mp4">
    </video>
</p>

## Getting Started

### 1. Download weights
Before running the project, download the following pretrained weights:
  - Pretrained weights of [DWPose](https://github.com/IDEA-Research/DWPose?tab=readme-ov-file#-dwpose-for-controlnet), including `yolox_l.onnx` and `dw-ll_ucoco_384.onnx`.
  - Pretrained weights of [AnimateAnyone](https://huggingface.co/patrolli/AnimateAnyone), including `denoising_unet.pth`, `reference_unet.pth`, `pose_guider.pth`, and `motion_module.pth`.
  - Pretrained weight of [MagicAnimate](https://huggingface.co/zcxu-eric/MagicAnimate/tree/main/appearance_encoder), including `appearance_encoder`.
  - Pretrained weight of [MagicPose](https://drive.google.com/drive/folders/1Ny5zkgo3aLVekCJTAga-D_XlMGpR1cj2?usp=sharing). including `model_state-110000.th`.
  - Pretrained weights of base models: [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5), [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse), and [sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers).

Organize the downloaded weights as follows:
```text
./pretrained_weights/
|-- DWPose
|   |-- dw-ll_ucoco_384.onnx
|   `-- yolox_l.onnx
|-- AnimateAnyone
|   |-- denoising_unet.pth
|   |-- reference_unet.pth
|   |-- pose_guider.pth
|   `-- motion_module.pth
|-- MagicAnimate
    |-- appearance_encoder
|       |-- diffusion_pytorch_model.safetensors
|       `-- config.json
|-- MagicPose
|   `-- model_state-110000.th
|-- stable-diffusion-v1-5
|-- sd-vae-ft-mse
`-- sd-image-variations-diffusers
```

### 2. Install environment
Set up the required environment using `conda`:
```bash
conda env create -f environment.yaml
conda activate dormant
```

### 3. Run protection
To protect a human image, run the following command:
```bash
python protect_image.py --ref_image_path ./inputs/000.png --output_dir ./outputs/
```

## Acknowledgments
Our code is built upon [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone), [MagicAnimate](https://github.com/magic-research/magic-animate), [MagicPose](https://github.com/Boese0601/MagicDance), [SDS](https://github.com/xavihart/Diff-Protect), and [Diff-JPEG](https://github.com/necla-ml/Diff-JPEG). We appreciate the authors for making their codes publicly available.
