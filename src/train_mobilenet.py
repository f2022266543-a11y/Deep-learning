"""
Training script for MobileNetV2 with transfer learning.
Lightweight model suitable for edge deployment.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import get_dataloaders
from src.utils import (
    get_device, train_one_epoch, validate, save_checkpoint,
    plot_training_curves, plot_confusion_matrix, save_training_log,
    count_parameters, EarlyStopping,
)


def build_mobilenetv2(num_classes, freeze_backbone=True):
    """Builds MobileNetV2 with pretrained weights and custom classifier."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        # Unfreeze last 3 inverted residual blocks
        for param in model.features[-3:].parameters():
            param.requires_grad = True

    # Replace classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes),
    )

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV2")
    parser.add_argument("--train_dir", type=str, default="datasets/PlantVillage/train")
    parser.add_argument("--val_dir", type=str, default="datasets/PlantVillage/val")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    print("Loading data...")
    train_loader, val_loader, num_classes = get_dataloaders(
        args.train_dir, args.val_dir, batch_size=args.batch_size
    )
    print(f"  Classes: {num_classes}")

    model = build_mobilenetv2(num_classes).to(device)
    count_parameters(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    early_stopping = EarlyStopping(patience=args.patience)

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
                "models/checkpoints/mobilenetv2_best.pt"
            )

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    torch.save(model.state_dict(), "models/checkpoints/mobilenetv2.pt")

    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        "results/logs/mobilenetv2_curves.png"
    )

    class_names = train_loader.dataset.classes
    plot_confusion_matrix(
        labels, preds, class_names,
        "results/confusion_matrix/mobilenetv2_cm.png"
    )

    save_training_log({
        "model": "MobileNetV2",
        "epochs": len(train_losses),
        "best_val_acc": best_val_acc,
    }, "results/logs/mobilenetv2_log.json")

    print(f"\nTraining complete! Best Val Acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
