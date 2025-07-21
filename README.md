# zeroFewShotSR


## How to Use the Code
Setup Environment:
```
pip install torch diffusers transformers torchvision scikit-image numpy pillow
```
Download SD v1.5: https://huggingface.co/runwayml/stable-diffusion-v1-5.
Download DINOv2 ViT-S/14: https://github.com/facebookresearch/dinov2.

## Prepare Test Image:
Use an LR image from DIV2K .
Example: Downsample an HR image to create an LR input (e.g., 128x128 from 512x512 for 4× SR).

## Run Zero-Shot SR:
Update lr_image_path.
```
python zeroShotSR.py.
```

## Run Few-Shot SR: 
```
python zeroFewShotSR.py.
```

## Evaluate Results:
If you have a ground-truth HR image, the code computes PSNR and SSIM using evaluate_sr.
Visually inspect the HR output for texture details and artifacts.