"""
DCGAN (Deep Convolutional Generative Adversarial Network)
for synthetic plant disease image generation.

Architecture follows the DCGAN paper (Radford et al., 2015)
with modifications for 64x64 plant disease images.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import get_gan_dataloader

# --- Hyperparameters ---
LATENT_DIM = 100
NGF = 64  # Generator feature map size
NDF = 64  # Discriminator feature map size
NC = 3    # Number of channels


def weights_init(m):
    """Custom weight initialization for DCGAN."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    DCGAN Generator.
    Maps latent vector z (100,) to image (3, 64, 64).

    Architecture:
        z → ConvTranspose(100→512) → ConvTranspose(512→256)
          → ConvTranspose(256→128) → ConvTranspose(128→64)
          → ConvTranspose(64→3) → Tanh
    """

    def __init__(self, latent_dim=LATENT_DIM, ngf=NGF, nc=NC):
        super().__init__()
        self.main = nn.Sequential(
            # Input: (latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State: (ngf*8, 4, 4)

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State: (ngf*4, 8, 8)

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State: (ngf*2, 16, 16)

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State: (ngf, 32, 32)

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # Output: (nc, 64, 64)
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    """
    DCGAN Discriminator.
    Maps image (3, 64, 64) to real/fake probability.

    Architecture:
        img → Conv(3→64) → Conv(64→128) → Conv(128→256)
            → Conv(256→512) → Conv(512→1) → Sigmoid
    """

    def __init__(self, ndf=NDF, nc=NC):
        super().__init__()
        self.main = nn.Sequential(
            # Input: (nc, 64, 64)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf, 32, 32)

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*2, 16, 16)

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*4, 8, 8)

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*8, 4, 4)

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # Output: (1, 1, 1)
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)


def train_dcgan(data_dir, epochs=100, lr=2e-4, beta1=0.5, batch_size=64,
                save_dir="results/dcgan", checkpoint_dir="models/checkpoints"):
    """
    Complete DCGAN training loop.

    Args:
        data_dir: Path to ImageFolder dataset.
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.
        beta1: Beta1 for Adam optimizer.
        batch_size: Training batch size.
        save_dir: Directory to save generated images.
        checkpoint_dir: Directory to save model checkpoints.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Data
    dataloader, num_classes = get_gan_dataloader(data_dir, batch_size=batch_size)

    # Models
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Fixed noise for tracking progress
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)

    # Labels
    real_label = 1.0
    fake_label = 0.0

    # Training
    G_losses = []
    D_losses = []

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (real_images, _) in enumerate(pbar):
            batch_size_curr = real_images.size(0)
            real_images = real_images.to(device)

            # ==========================================
            # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            # ==========================================
            netD.zero_grad()

            # Real batch
            labels = torch.full((batch_size_curr,), real_label, device=device)
            output = netD(real_images)
            errD_real = criterion(output, labels)
            errD_real.backward()
            D_x = output.mean().item()

            # Fake batch
            noise = torch.randn(batch_size_curr, LATENT_DIM, 1, 1, device=device)
            fake = netG(noise)
            labels.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, labels)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            # ==========================================
            # Train Generator: max log(D(G(z)))
            # ==========================================
            netG.zero_grad()
            labels.fill_(real_label)
            output = netD(fake)
            errG = criterion(output, labels)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            pbar.set_postfix(
                D_loss=f"{errD.item():.4f}",
                G_loss=f"{errG.item():.4f}",
                D_x=f"{D_x:.4f}",
                D_G=f"{D_G_z1:.4f}/{D_G_z2:.4f}",
            )

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Save sample images
        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                fake_images = netG(fixed_noise).detach().cpu()
            vutils.save_image(
                fake_images, f"{save_dir}/epoch_{epoch+1:03d}.png",
                normalize=True, nrow=8,
            )
            print(f"  Saved samples: {save_dir}/epoch_{epoch+1:03d}.png")

    # Save models
    torch.save(netG.state_dict(), f"{checkpoint_dir}/dcgan_generator.pt")
    torch.save(netD.state_dict(), f"{checkpoint_dir}/dcgan_discriminator.pt")

    # Plot losses
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(G_losses, label="Generator", linewidth=2)
    ax.plot(D_losses, label="Discriminator", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("DCGAN Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("DCGAN training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DCGAN")
    parser.add_argument("--data_dir", type=str, default="datasets/PlantVillage/train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    train_dcgan(args.data_dir, args.epochs, args.lr, args.batch_size)
