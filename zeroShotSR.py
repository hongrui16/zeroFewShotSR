import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from transformers import Dinov2Model
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path


# 6. Evaluation Function (PSNR and SSIM)
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


'''
pip install torch diffusers transformers torchvision scikit-image numpy pillow

'''

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Pre-trained Models
# Stable Diffusion v1.5
sd_pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to(device)
sd_pipeline.safety_checker = None  # Disable for research purposes
vae = sd_pipeline.vae
unet = sd_pipeline.unet
scheduler = sd_pipeline.scheduler

# DINOv2 ViT-S/14
dinov2 = Dinov2Model.from_pretrained("facebook/dinov2-small").to(device)
dinov2.eval()

# 2. Context Mapping MLP
class ContextMappingMLP(nn.Module):
    def __init__(self, input_dim=384, output_dim=512):
        super(ContextMappingMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.mlp(x)

context_mlp = ContextMappingMLP().to(device)

# 3. Dynamic Latent Modulation Network
class LatentModulationNetwork(nn.Module):
    def __init__(self, context_dim=512, attention_dim=512):
        super(LatentModulationNetwork, self).__init__()
        self.modulation = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, attention_dim)
        )

    def forward(self, context_vector, attention_weights):
        modulation = self.modulation(context_vector)
        modulated_weights = attention_weights * modulation.unsqueeze(-1).unsqueeze(-1)
        return modulated_weights

modulation_network = LatentModulationNetwork().to(device)

# 4. Image Preprocessing
dinov2_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

sd_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 5. Zero-Shot SR Function
def zero_shot_sr(lr_image_path, scale_factor=4, num_steps=50):
    # Load and preprocess LR image
    lr_image = Image.open(lr_image_path).convert("RGB")
    lr_tensor_dinov2 = dinov2_transform(lr_image).unsqueeze(0).to(device)
    lr_tensor_sd = sd_transform(lr_image).unsqueeze(0).to(device).half()

    # Extract context features with DINOv2
    with torch.no_grad():
        dinov2_features = dinov2(lr_tensor_dinov2).last_hidden_state  # [1, num_patches, 384]
        context_vector = dinov2_features.mean(dim=1)  # [1, 384]

    # Map to SD latent space
    latent_context = context_mlp(context_vector)  # [1, 512]

    # Encode LR image to SD latent
    with torch.no_grad():
        lr_latent = vae.encode(lr_tensor_sd).latent_dist.sample() * vae.config.scaling_factor

    # Initialize noise
    batch_size, channels, height, width = lr_latent.shape
    hr_height, hr_width = height * scale_factor, width * scale_factor
    latent = torch.randn(batch_size, channels, hr_height, hr_width).to(device)

    # Modulate attention layers
    def modulated_attention_forward(self, hidden_states, context=None, **kwargs):
        attention_weights = self.attn1(hidden_states, context=context)
        modulated_weights = modulation_network(latent_context, attention_weights)
        return modulated_weights

    target_attn_layer = unet.up_blocks[1].attentions[0]
    target_attn_layer.forward = modulated_attention_forward.__get__(
        target_attn_layer, target_attn_layer.__class__
    )

    # Denoising loop (multi-step for initial testing)
    scheduler.set_timesteps(num_steps)
    for t in scheduler.timesteps:
        with torch.no_grad():
            noisy_latent = scheduler.add_noise(latent, torch.randn_like(latent), t)
            latent = unet(noisy_latent, t, encoder_hidden_states=latent_context).sample
            latent = scheduler.step(latent, t, latent).prev_sample

    # Decode to HR image
    with torch.no_grad():
        hr_image = vae.decode(latent / vae.config.scaling_factor).sample
    hr_image = (hr_image / 2 + 0.5).clamp(0, 1)
    hr_image = hr_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    hr_image = Image.fromarray((hr_image * 255).astype(np.uint8))

    return hr_image



def evaluate_sr(hr_image, gt_image_path):
    gt_image = Image.open(gt_image_path).convert("RGB").resize(hr_image.size)
    gt_array = np.array(gt_image)
    hr_array = np.array(hr_image)

    psnr = peak_signal_noise_ratio(gt_array, hr_array, data_range=255)
    ssim = structural_similarity(gt_array, hr_array, multichannel=True, channel_axis=2)
    return psnr, ssim

# 7. Example Usage
if __name__ == "__main__":
    # Paths (replace with your image paths)
    lr_image_path = "path_to_lr_image.jpg"  # Low-resolution input
    gt_image_path = "path_to_hr_image.jpg"  # Ground-truth HR (for evaluation)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Run zero-shot SR
    hr_image = zero_shot_sr(lr_image_path, scale_factor=4, num_steps=50)
    output_path = output_dir / "hr_output.jpg"
    hr_image.save(output_path)

    # Evaluate (if ground-truth HR is available)
    if gt_image_path:
        psnr, ssim = evaluate_sr(hr_image, gt_image_path)
        print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")