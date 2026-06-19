"""
Denoising Diffusion Probabilistic Model (DDPM) for plant disease image generation.

Implements the Ho et al., 2020 DDPM paper with a U-Net backbone.
Generates high-quality 64x64 plant disease images through iterative denoising.
"""

import os
import sys
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import get_gan_dataloader


# ============================================================
# Noise Scheduler
# ============================================================

class NoiseScheduler:
    """Linear noise schedule for DDPM."""

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)

        # Pre-compute useful quantities
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alphas)

        # Posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        )

    def add_noise(self, x0, t, noise=None):
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def sample_timesteps(self, batch_size):
        """Sample random timesteps for training."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)


# ============================================================
# U-Net Components
# ============================================================

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for timestep encoding."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ResBlock(nn.Module):
    """Residual block with timestep embedding injection."""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        h = h + self.time_mlp(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = torch.einsum("bci,bcj->bij", q, k) * (C ** -0.5)
        attn = attn.softmax(dim=-1)

        out = torch.einsum("bij,bcj->bci", attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ============================================================
# U-Net
# ============================================================

class UNet(nn.Module):
    """
    U-Net architecture for DDPM noise prediction.

    Input: (batch, 3, 64, 64) + timestep
    Output: (batch, 3, 64, 64) predicted noise
    """

    def __init__(self, in_ch=3, base_ch=64, time_dim=256, ch_mults=(1, 2, 4, 8)):
        super().__init__()

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channels = [base_ch]
        ch = base_ch

        for mult in ch_mults:
            out_ch = base_ch * mult
            self.encoder_blocks.append(nn.ModuleList([
                ResBlock(ch, out_ch, time_dim),
                ResBlock(out_ch, out_ch, time_dim),
                AttentionBlock(out_ch) if mult >= 4 else nn.Identity(),
            ]))
            self.downsamples.append(Downsample(out_ch))
            channels.append(out_ch)
            ch = out_ch

        # Bottleneck
        self.mid_block1 = ResBlock(ch, ch, time_dim)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResBlock(ch, ch, time_dim)

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for mult in reversed(ch_mults):
            out_ch = base_ch * mult
            self.upsamples.append(Upsample(ch))
            skip_ch = channels.pop()
            self.decoder_blocks.append(nn.ModuleList([
                ResBlock(ch + skip_ch, out_ch, time_dim),
                ResBlock(out_ch, out_ch, time_dim),
                AttentionBlock(out_ch) if mult >= 4 else nn.Identity(),
            ]))
            ch = out_ch

        # Final output
        self.final = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_ch, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        x = self.init_conv(x)

        # Encoder with skip connections
        skips = [x]
        for blocks, downsample in zip(self.encoder_blocks, self.downsamples):
            res1, res2, attn = blocks
            x = res1(x, t_emb)
            x = res2(x, t_emb)
            x = attn(x) if not isinstance(attn, nn.Identity) else x
            skips.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # Decoder
        for blocks, upsample in zip(self.decoder_blocks, self.upsamples):
            x = upsample(x)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            res1, res2, attn = blocks
            x = res1(x, t_emb)
            x = res2(x, t_emb)
            x = attn(x) if not isinstance(attn, nn.Identity) else x

        return self.final(x)


# ============================================================
# DDPM Sampling
# ============================================================

@torch.no_grad()
def ddpm_sample(model, scheduler, shape, device):
    """
    Generates images using DDPM reverse process.

    Args:
        model: Trained UNet noise predictor.
        scheduler: NoiseScheduler instance.
        shape: Output shape (batch, channels, height, width).
        device: Torch device.

    Returns:
        Generated images tensor.
    """
    model.eval()
    x = torch.randn(shape, device=device)

    for t in tqdm(reversed(range(scheduler.num_timesteps)), desc="Sampling", total=scheduler.num_timesteps):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

        # Predict noise
        predicted_noise = model(x, t_batch)

        # Compute x_{t-1}
        alpha = scheduler.alphas[t]
        alpha_cumprod = scheduler.alpha_cumprod[t]
        beta = scheduler.betas[t]

        # Mean
        mean = scheduler.sqrt_recip_alpha[t] * (
            x - beta / scheduler.sqrt_one_minus_alpha_cumprod[t] * predicted_noise
        )

        # Add noise (except at t=0)
        if t > 0:
            noise = torch.randn_like(x)
            variance = torch.sqrt(scheduler.posterior_variance[t])
            x = mean + variance * noise
        else:
            x = mean

    return x


# ============================================================
# Training Loop
# ============================================================

def train_ddpm(data_dir, epochs=200, lr=2e-4, batch_size=32, num_timesteps=1000,
               save_dir="results/ddpm", checkpoint_dir="models/checkpoints"):
    """
    Complete DDPM training loop.

    Args:
        data_dir: Path to ImageFolder dataset.
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Training batch size.
        num_timesteps: Number of diffusion timesteps.
        save_dir: Directory to save generated images.
        checkpoint_dir: Directory for model checkpoints.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Data
    dataloader, num_classes = get_gan_dataloader(data_dir, batch_size=batch_size)

    # Model and scheduler
    model = UNet(in_ch=3, base_ch=64).to(device)
    scheduler = NoiseScheduler(num_timesteps=num_timesteps, device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  UNet parameters: {total_params:,}")

    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, _ in pbar:
            images = images.to(device)

            # Sample random timesteps
            t = scheduler.sample_timesteps(images.size(0))

            # Forward diffusion
            noise = torch.randn_like(images)
            noisy_images = scheduler.add_noise(images, t, noise)

            # Predict noise
            predicted_noise = model(noisy_images, t)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}: Avg Loss = {avg_loss:.6f}")

        # Generate samples periodically
        if (epoch + 1) % 20 == 0 or epoch == 0:
            samples = ddpm_sample(model, scheduler, (16, 3, 64, 64), device)
            samples = (samples.clamp(-1, 1) + 1) / 2  # Scale to [0, 1]
            vutils.save_image(
                samples, f"{save_dir}/epoch_{epoch+1:03d}.png",
                nrow=4, normalize=False,
            )
            print(f"  Saved samples: {save_dir}/epoch_{epoch+1:03d}.png")

    # Save model
    torch.save(model.state_dict(), f"{checkpoint_dir}/ddpm_unet.pt")

    # Plot loss
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(losses, linewidth=2, color="#2196F3")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("DDPM Training Loss")
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("DDPM training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPM")
    parser.add_argument("--data_dir", type=str, default="datasets/PlantVillage/train")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=1000)
    args = parser.parse_args()

    train_ddpm(args.data_dir, args.epochs, args.lr, args.batch_size, args.timesteps)
