"""
Training script for Custom CNN on PlantVillage/PlantDoc dataset.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import get_dataloaders
from models.custom_cnn import CustomCNN
from src.utils import (
    get_device, train_one_epoch, validate, save_checkpoint,
    plot_training_curves, plot_confusion_matrix, save_training_log,
    count_parameters, EarlyStopping,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Custom CNN")
    parser.add_argument("--train_dir", type=str, default="datasets/PlantVillage/train")
    parser.add_argument("--val_dir", type=str, default="datasets/PlantVillage/val")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    # Data
    print("Loading data...")
    train_loader, val_loader, num_classes = get_dataloaders(
        args.train_dir, args.val_dir, batch_size=args.batch_size
    )
    print(f"  Classes: {num_classes}")

    # Model
    model = CustomCNN(num_classes).to(device)
    count_parameters(model)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    early_stopping = EarlyStopping(patience=args.patience)

    # Tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    # Training loop
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

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                "models/checkpoints/custom_cnn_best.pt"
            )

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save final model
    torch.save(model.state_dict(), "models/checkpoints/custom_cnn.pt")

    # Plots
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        "results/logs/custom_cnn_curves.png"
    )

    class_names = train_loader.dataset.classes
    plot_confusion_matrix(
        labels, preds, class_names,
        "results/confusion_matrix/custom_cnn_cm.png"
    )

    # Save log
    save_training_log({
        "model": "CustomCNN",
        "epochs": len(train_losses),
        "best_val_acc": best_val_acc,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
    }, "results/logs/custom_cnn_log.json")

    print(f"\nTraining complete! Best Val Acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
