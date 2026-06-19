"""
Vision Transformer (ViT) training using timm library.
Includes ViT-B/16, DeiT-Small, and Swin-Tiny models.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import timm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import get_dataloaders
from src.utils import (
    get_device, train_one_epoch, validate, save_checkpoint,
    plot_training_curves, plot_confusion_matrix, save_training_log,
    count_parameters, EarlyStopping,
)


# ============================================================
# Model Builders
# ============================================================

def build_vit(num_classes, model_name="vit_base_patch16_224", pretrained=True):
    """
    Builds a Vision Transformer using timm.

    Args:
        num_classes: Number of output classes.
        model_name: timm model name.
        pretrained: Whether to use pretrained weights.

    Returns:
        Configured ViT model.
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def build_deit(num_classes, pretrained=True):
    """Builds DeiT-Small model."""
    model = timm.create_model("deit_small_patch16_224", pretrained=pretrained, num_classes=num_classes)
    return model


def build_swin(num_classes, pretrained=True):
    """Builds Swin-Tiny Transformer model."""
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=num_classes)
    return model


# ============================================================
# Hybrid CNN-ViT Model
# ============================================================

class HybridCNNViT(nn.Module):
    """
    Hybrid CNN-ViT model.

    Uses CNN backbone for feature extraction and ViT for classification.
    CNN extracts spatial features → projected to patch embeddings → ViT encoder.
    """

    def __init__(self, num_classes, cnn_features=256, vit_dim=512,
                 num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, cnn_features, 3, padding=1),
            nn.BatchNorm2d(cnn_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # After CNN: (batch, cnn_features, 28, 28) for 224x224 input

        # Project CNN features to ViT dimension
        self.patch_embed = nn.Linear(cnn_features, vit_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, vit_dim))

        # Positional embedding: 28*28 patches + 1 CLS = 785 tokens
        self.pos_embed = nn.Parameter(torch.randn(1, 28 * 28 + 1, vit_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vit_dim,
            nhead=num_heads,
            dim_feedforward=vit_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.norm = nn.LayerNorm(vit_dim)
        self.head = nn.Sequential(
            nn.Linear(vit_dim, vit_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(vit_dim // 2, num_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)

        # CNN feature extraction
        x = self.cnn(x)  # (B, cnn_features, 28, 28)
        x = x.flatten(2).transpose(1, 2)  # (B, 784, cnn_features)

        # Project to ViT dimension
        x = self.patch_embed(x)  # (B, 784, vit_dim)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 785, vit_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer encoder
        x = self.transformer(x)

        # Classification from CLS token
        x = self.norm(x[:, 0])
        x = self.head(x)

        return x


# ============================================================
# Training Pipeline
# ============================================================

MODEL_BUILDERS = {
    "vit": build_vit,
    "deit": build_deit,
    "swin": build_swin,
    "hybrid": lambda nc, **kw: HybridCNNViT(nc),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Vision Transformer Models")
    parser.add_argument("--model", type=str, default="vit",
                        choices=["vit", "deit", "swin", "hybrid"])
    parser.add_argument("--train_dir", type=str, default="datasets/PlantVillage/train")
    parser.add_argument("--val_dir", type=str, default="datasets/PlantVillage/val")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    print(f"\n{'='*60}")
    print(f"Training {args.model.upper()} Transformer")
    print(f"{'='*60}")

    # Data
    print("Loading data...")
    train_loader, val_loader, num_classes = get_dataloaders(
        args.train_dir, args.val_dir, batch_size=args.batch_size
    )
    print(f"  Classes: {num_classes}")

    # Model
    builder = MODEL_BUILDERS[args.model]
    if args.model == "hybrid":
        model = builder(num_classes)
    else:
        model = builder(num_classes, pretrained=True)
    model = model.to(device)
    count_parameters(model)

    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2
    )
    early_stopping = EarlyStopping(patience=args.patience)

    # Tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scheduler
        )
        val_loss, val_acc, preds, labels = validate(
            model, val_loader, criterion, device, epoch
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                f"models/checkpoints/{args.model}_best.pt"
            )

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save final
    torch.save(model.state_dict(), f"models/checkpoints/{args.model}.pt")

    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        f"results/logs/{args.model}_curves.png"
    )

    class_names = train_loader.dataset.classes
    plot_confusion_matrix(
        labels, preds, class_names,
        f"results/confusion_matrix/{args.model}_cm.png"
    )

    save_training_log({
        "model": args.model.upper(),
        "epochs": len(train_losses),
        "best_val_acc": best_val_acc,
    }, f"results/logs/{args.model}_log.json")

    print(f"\n{args.model.upper()} Training complete! Best Val Acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
