"""
Training script for ResNet50 with transfer learning.
Freezes early layers, fine-tunes layer4 and classifier head.
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


def build_resnet50(num_classes, freeze_backbone=True):
    """Builds ResNet50 with pretrained weights and custom classifier."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze layer4 for fine-tuning
        for param in model.layer4.parameters():
            param.requires_grad = True

    # Replace classifier head
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, num_classes),
    )

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet50")
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

    model = build_resnet50(num_classes).to(device)
    count_parameters(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
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
                "models/checkpoints/resnet50_best.pt"
            )

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    torch.save(model.state_dict(), "models/checkpoints/resnet50.pt")

    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        "results/logs/resnet50_curves.png"
    )

    class_names = train_loader.dataset.classes
    plot_confusion_matrix(
        labels, preds, class_names,
        "results/confusion_matrix/resnet50_cm.png"
    )

    save_training_log({
        "model": "ResNet50",
        "epochs": len(train_losses),
        "best_val_acc": best_val_acc,
    }, "results/logs/resnet50_log.json")

    print(f"\nTraining complete! Best Val Acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
