"""
Comprehensive evaluation module.
Computes accuracy, F1, precision, recall, confusion matrix, and per-class metrics.
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import get_dataloaders


def evaluate_model(model, dataloader, device, class_names=None):
    """
    Evaluates a model on a given dataloader.

    Args:
        model: PyTorch model.
        dataloader: Validation/test DataLoader.
        device: Torch device.
        class_names: List of class name strings.

    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    # Core metrics
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Per-class report
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
    )

    results = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    return results


def print_results(results):
    """Pretty-prints evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy:          {results['accuracy']*100:.2f}%")
    print(f"  F1 (Macro):        {results['f1_macro']*100:.2f}%")
    print(f"  F1 (Weighted):     {results['f1_weighted']*100:.2f}%")
    print(f"  Precision (Macro): {results['precision_macro']*100:.2f}%")
    print(f"  Recall (Macro):    {results['recall_macro']*100:.2f}%")
    print("=" * 60)


def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    """Generates and saves a confusion matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig_size = max(10, len(class_names) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        square=True,
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")


def save_results(results, save_path):
    """Saves evaluation results as JSON."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved: {save_path}")


def compare_models(results_dict, save_path="results/logs/model_comparison.png"):
    """
    Creates a comparison bar chart of model accuracies and F1 scores.

    Args:
        results_dict: Dict mapping model_name -> results dict.
        save_path: Path to save comparison chart.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    models = list(results_dict.keys())
    accs = [results_dict[m]["accuracy"] * 100 for m in models]
    f1s = [results_dict[m]["f1_macro"] * 100 for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, f1s, width, label="F1 Score", color="#FF9800", alpha=0.85)

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(80, 100)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars1:
        ax.annotate(
            f"{bar.get_height():.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
        )
    for bar in bars2:
        ax.annotate(
            f"{bar.get_height():.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Model comparison chart saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["custom_cnn", "resnet50", "efficientnet_b3", "mobilenetv2"])
    parser.add_argument("--val_dir", type=str, default="datasets/PlantVillage/val")
    parser.add_argument("--train_dir", type=str, default="datasets/PlantVillage/train")
    args = parser.parse_args()

    from src.utils import get_device

    device = get_device()

    # Load data to get class info
    train_loader, val_loader, num_classes = get_dataloaders(
        args.train_dir, args.val_dir
    )
    class_names = train_loader.dataset.classes

    # Build model
    if args.model_type == "custom_cnn":
        from models.custom_cnn import CustomCNN
        model = CustomCNN(num_classes)
    elif args.model_type == "resnet50":
        from src.train_resnet50 import build_resnet50
        model = build_resnet50(num_classes, freeze_backbone=False)
    elif args.model_type == "efficientnet_b3":
        from src.train_efficientnet import build_efficientnet_b3
        model = build_efficientnet_b3(num_classes, freeze_backbone=False)
    elif args.model_type == "mobilenetv2":
        from src.train_mobilenet import build_mobilenetv2
        model = build_mobilenetv2(num_classes, freeze_backbone=False)

    # Load weights
    checkpoint = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)

    # Evaluate
    results = evaluate_model(model, val_loader, device, class_names)
    print_results(results)

    # Save outputs
    plot_confusion_matrix(
        np.array(results["confusion_matrix"]),
        class_names,
        f"results/confusion_matrix/{args.model_type}_eval_cm.png",
    )
    save_results(results, f"results/logs/{args.model_type}_eval_results.json")
