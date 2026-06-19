"""
Ensemble system for combining multiple model predictions.
Supports average softmax, weighted ensemble, and majority voting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


class EnsemblePredictor:
    """
    Ensemble prediction system that combines outputs from multiple models.

    Supports three strategies:
    1. Average Softmax: Equal-weight averaging of softmax probabilities.
    2. Weighted Ensemble: Weighted averaging with custom per-model weights.
    3. Majority Voting: Hard voting based on individual model predictions.
    """

    def __init__(self, models: List[nn.Module], device: str = "cpu",
                 weights: Optional[List[float]] = None):
        """
        Args:
            models: List of trained PyTorch models.
            device: Torch device.
            weights: Optional list of weights for weighted ensemble.
                     Must sum to 1.0. If None, uses equal weights.
        """
        self.models = models
        self.device = device

        if weights is not None:
            assert len(weights) == len(models), "Weights must match number of models"
            assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
            self.weights = weights
        else:
            self.weights = [1.0 / len(models)] * len(models)

        # Set all models to eval mode
        for model in self.models:
            model.eval()

    def _get_outputs(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Gets raw outputs from all models."""
        outputs = []
        with torch.no_grad():
            for model in self.models:
                output = model(images.to(self.device))
                outputs.append(output)
        return outputs

    def average_softmax(self, images: torch.Tensor) -> torch.Tensor:
        """
        Average Softmax Ensemble.
        Averages softmax probabilities across all models equally.

        Args:
            images: Input tensor (batch, C, H, W).

        Returns:
            Averaged probability tensor (batch, num_classes).
        """
        outputs = self._get_outputs(images)
        probs = [F.softmax(o, dim=1) for o in outputs]
        return sum(probs) / len(probs)

    def weighted_ensemble(self, images: torch.Tensor,
                          weights: Optional[List[float]] = None) -> torch.Tensor:
        """
        Weighted Ensemble.
        Weighted average of softmax probabilities.

        Args:
            images: Input tensor (batch, C, H, W).
            weights: Optional override weights. Uses init weights if None.

        Returns:
            Weighted probability tensor (batch, num_classes).
        """
        w = weights if weights is not None else self.weights
        outputs = self._get_outputs(images)

        final = torch.zeros_like(F.softmax(outputs[0], dim=1))
        for out, weight in zip(outputs, w):
            final += F.softmax(out, dim=1) * weight

        return final

    def majority_vote(self, images: torch.Tensor) -> torch.Tensor:
        """
        Majority Voting Ensemble.
        Each model votes for a class; the most voted class wins.

        Args:
            images: Input tensor (batch, C, H, W).

        Returns:
            Predicted class indices (batch,).
        """
        outputs = self._get_outputs(images)
        predictions = [o.argmax(dim=1) for o in outputs]
        stacked = torch.stack(predictions, dim=0)  # (num_models, batch)

        # Mode (most common) across models
        votes, _ = torch.mode(stacked, dim=0)
        return votes

    def predict(self, images: torch.Tensor, strategy: str = "weighted") -> dict:
        """
        Make ensemble predictions with specified strategy.

        Args:
            images: Input tensor (batch, C, H, W).
            strategy: One of 'average', 'weighted', or 'vote'.

        Returns:
            Dictionary with 'predictions', 'probabilities', and 'confidence'.
        """
        if strategy == "average":
            probs = self.average_softmax(images)
        elif strategy == "weighted":
            probs = self.weighted_ensemble(images)
        elif strategy == "vote":
            preds = self.majority_vote(images)
            return {
                "predictions": preds,
                "probabilities": None,
                "confidence": None,
            }
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        confidence, predictions = torch.max(probs, dim=1)

        return {
            "predictions": predictions,
            "probabilities": probs,
            "confidence": confidence,
        }


def evaluate_ensemble(ensemble, dataloader, device, strategy="weighted"):
    """
    Evaluates ensemble on a validation/test set.

    Args:
        ensemble: EnsemblePredictor instance.
        dataloader: DataLoader for evaluation.
        device: Torch device.
        strategy: Ensemble strategy to use.

    Returns:
        accuracy, all_preds, all_labels
    """
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        result = ensemble.predict(images, strategy=strategy)
        all_preds.extend(result["predictions"].cpu().numpy())
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean() * 100

    return accuracy, all_preds, all_labels


# ============================================================
# Standalone usage example
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    print("Ensemble System")
    print("=" * 40)
    print("Supported strategies:")
    print("  1. average  - Equal-weight softmax averaging")
    print("  2. weighted - Weighted softmax averaging")
    print("  3. vote     - Majority hard voting")
    print()
    print("Default weights for 3-model ensemble:")
    print("  MobileNetV2:    0.20")
    print("  EfficientNet-B3: 0.30")
    print("  Hybrid CNN-ViT:  0.50")
    print()
    print("Usage:")
    print("  from src.ensemble import EnsemblePredictor")
    print("  ensemble = EnsemblePredictor(models, weights=[0.20, 0.30, 0.50])")
    print("  result = ensemble.predict(images, strategy='weighted')")
