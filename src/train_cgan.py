"""
Conditional GAN (cGAN) for class-conditional plant disease image generation.

Extends DCGAN by conditioning both Generator and Discriminator on class labels,
enabling targeted generation of specific disease types.
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
NGF = 64
NDF = 64
NC = 3
EMBED_DIM = 50  # Class embedding dimension


class ConditionalGenerator(nn.Module):
    """
    Conditional DCGAN Generator.
    Maps (z, class_label) to image (3, 64, 64).

    The class label is embedded and concatenated with the noise vector.
    """

    def __init__(self, num_classes, latent_dim=LATENT_DIM, ngf=NGF, nc=NC, embed_dim=EMBED_DIM):
        super().__init__()

        self.label_embed = nn.Embedding(num_classes, embed_dim)

        input_dim = latent_dim + embed_dim

        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        # z: (batch, latent_dim, 1, 1)
        # labels: (batch,) integer labels
        label_embed = self.label_embed(labels).unsqueeze(2).unsqueeze(3)  # (batch, embed_dim, 1, 1)
        z_concat = torch.cat([z, label_embed], dim=1)
        return self.main(z_concat)


class ConditionalDiscriminator(nn.Module):
    """
    Conditional DCGAN Discriminator.
    Maps (image, class_label) to real/fake probability.

    The class label is embedded and spatially broadcast to match image dimensions.
    """

    def __init__(self, num_classes, ndf=NDF, nc=NC, embed_dim=EMBED_DIM, image_size=64):
        super().__init__()

        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.label_fc = nn.Linear(embed_dim, image_size * image_size)
        self.image_size = image_size

        self.main = nn.Sequential(
            # Input: (nc + 1, 64, 64) -- image + label channel
            nn.Conv2d(nc + 1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, images, labels):
        # Create label channel
        label_embed = self.label_embed(labels)
        label_map = self.label_fc(label_embed)
        label_map = label_map.view(-1, 1, self.image_size, self.image_size)

        # Concatenate with image
        x = torch.cat([images, label_map], dim=1)
        return self.main(x).view(-1, 1).squeeze(1)


def train_conditional_gan(data_dir, epochs=100, lr=2e-4, beta1=0.5,
                           batch_size=64, save_dir="results/cgan",
                           checkpoint_dir="models/checkpoints"):
    """
    Complete Conditional GAN training loop.

    Args:
        data_dir: Path to ImageFolder dataset.
        epochs: Number of training epochs.
        lr: Learning rate.
        beta1: Adam beta1.
        batch_size: Training batch size.
        save_dir: Directory to save generated images.
        checkpoint_dir: Directory for model checkpoints.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Data
    dataloader, num_classes = get_gan_dataloader(data_dir, batch_size=batch_size)
    print(f"  Number of classes: {num_classes}")

    # Models
    netG = ConditionalGenerator(num_classes).to(device)
    netD = ConditionalDiscriminator(num_classes).to(device)

    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Fixed noise and labels for tracking progress
    fixed_noise = torch.randn(num_classes * 4, LATENT_DIM, 1, 1, device=device)
    fixed_labels = torch.arange(num_classes, device=device).repeat(4)

    G_losses = []
    D_losses = []

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for real_images, real_labels in pbar:
            batch_size_curr = real_images.size(0)
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)

            # === Train Discriminator ===
            netD.zero_grad()

            # Real
            labels_real = torch.ones(batch_size_curr, device=device)
            output = netD(real_images, real_labels)
            errD_real = criterion(output, labels_real)
            errD_real.backward()

            # Fake
            noise = torch.randn(batch_size_curr, LATENT_DIM, 1, 1, device=device)
            fake_class_labels = torch.randint(0, num_classes, (batch_size_curr,), device=device)
            fake = netG(noise, fake_class_labels)
            labels_fake = torch.zeros(batch_size_curr, device=device)
            output = netD(fake.detach(), fake_class_labels)
            errD_fake = criterion(output, labels_fake)
            errD_fake.backward()

            errD = errD_real + errD_fake
            optimizerD.step()

            # === Train Generator ===
            netG.zero_grad()
            output = netD(fake, fake_class_labels)
            errG = criterion(output, labels_real)
            errG.backward()
            optimizerG.step()

            pbar.set_postfix(
                D_loss=f"{errD.item():.4f}",
                G_loss=f"{errG.item():.4f}",
            )

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Save samples
        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                fake_images = netG(fixed_noise, fixed_labels).detach().cpu()
            vutils.save_image(
                fake_images, f"{save_dir}/epoch_{epoch+1:03d}.png",
                normalize=True, nrow=num_classes,
            )

    # Save models
    torch.save(netG.state_dict(), f"{checkpoint_dir}/cgan_generator.pt")
    torch.save(netD.state_dict(), f"{checkpoint_dir}/cgan_discriminator.pt")

    # Plot losses
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(G_losses, label="Generator", linewidth=2)
    ax.plot(D_losses, label="Discriminator", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Conditional GAN Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Conditional GAN training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Conditional GAN")
    parser.add_argument("--data_dir", type=str, default="datasets/PlantVillage/train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    train_conditional_gan(args.data_dir, args.epochs, args.lr, args.batch_size)
