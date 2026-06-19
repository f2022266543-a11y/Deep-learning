"""
Grad-CAM and Attention visualization for model explainability.
Supports CNN models (Grad-CAM) and Vision Transformers (Attention Rollout).
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# Grad-CAM Implementation
# ============================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Generates visual explanations for CNN-based models by computing
    the gradient of the target class score with respect to feature maps.
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: Trained CNN model.
            target_layer: The convolutional layer to visualize.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generates Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor (1, C, H, W).
            target_class: Target class index. If None, uses predicted class.

        Returns:
            Normalized heatmap as numpy array (H, W).
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Global average pool the gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, target_class


def apply_heatmap(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlays Grad-CAM heatmap on the original image.

    Args:
        image: Original image as numpy array (H, W, 3) in [0, 255].
        heatmap: Grad-CAM heatmap (H, W) in [0, 1].
        alpha: Blending factor.
        colormap: OpenCV colormap.

    Returns:
        Blended image as numpy array.
    """
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    blended = np.float32(heatmap_colored) * alpha + np.float32(image) * (1 - alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended


# ============================================================
# Attention Rollout (for Vision Transformers)
# ============================================================

def attention_rollout(attentions, discard_ratio=0.1, head_fusion="mean"):
    """
    Computes attention rollout from a list of attention matrices.

    Args:
        attentions: List of attention matrices from ViT layers.
                   Each has shape (batch, heads, tokens, tokens).
        discard_ratio: Fraction of lowest attention weights to discard.
        head_fusion: How to combine heads - 'mean', 'max', or 'min'.

    Returns:
        Rollout attention map (tokens,) for [CLS] token.
    """
    result = torch.eye(attentions[0].size(-1))

    with torch.no_grad():
        for attention in attentions:
            # Fuse heads
            if head_fusion == "mean":
                attention_fused = attention.mean(dim=1)
            elif head_fusion == "max":
                attention_fused = attention.max(dim=1)[0]
            elif head_fusion == "min":
                attention_fused = attention.min(dim=1)[0]
            else:
                raise ValueError(f"Unknown head_fusion: {head_fusion}")

            attention_fused = attention_fused.squeeze(0)  # (tokens, tokens)

            # Discard low attention
            flat = attention_fused.view(-1)
            threshold = flat.kthvalue(int(flat.size(0) * discard_ratio))[0]
            attention_fused = torch.where(
                attention_fused > threshold,
                attention_fused,
                torch.zeros_like(attention_fused),
            )

            # Add identity (residual connection)
            I = torch.eye(attention_fused.size(-1))
            a = (attention_fused + I) / 2

            # Normalize rows
            a = a / a.sum(dim=-1, keepdim=True)

            result = torch.matmul(a, result)

    # Return CLS token attention to all other tokens
    mask = result[0, 1:]  # Exclude CLS self-attention
    mask = mask / mask.max()

    return mask.numpy()


# ============================================================
# Entropy Analysis
# ============================================================

def attention_entropy(attention_weights):
    """
    Computes entropy of attention distributions per head.

    Args:
        attention_weights: Attention matrix (batch, heads, tokens, tokens).

    Returns:
        Mean entropy per head.
    """
    # Clamp to avoid log(0)
    attn = attention_weights.clamp(min=1e-8)
    entropy = -(attn * attn.log()).sum(dim=-1)  # (batch, heads, tokens)
    mean_entropy = entropy.mean(dim=[0, 2])  # (heads,)
    return mean_entropy


# ============================================================
# Visualization Utilities
# ============================================================

def visualize_gradcam(image_path, model, target_layer, transform, device,
                      save_path=None, class_names=None):
    """
    Complete Grad-CAM visualization pipeline.

    Args:
        image_path: Path to input image.
        model: Trained model.
        target_layer: Target convolutional layer.
        transform: Image preprocessing transform.
        device: Torch device.
        save_path: Optional path to save visualization.
        class_names: Optional list of class names.
    """
    # Load and preprocess image
    raw_image = Image.open(image_path).convert("RGB")
    raw_np = np.array(raw_image.resize((224, 224)))
    input_tensor = transform(raw_image).unsqueeze(0).to(device)

    # Generate Grad-CAM
    gradcam = GradCAM(model, target_layer)
    heatmap, pred_class = gradcam.generate(input_tensor)

    # Apply heatmap
    blended = apply_heatmap(raw_np, heatmap)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(raw_np)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")

    pred_label = class_names[pred_class] if class_names else str(pred_class)
    axes[2].imshow(blended)
    axes[2].set_title(f"Overlay (Pred: {pred_label})", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Grad-CAM saved: {save_path}")

    plt.close()


def visualize_attention_rollout(image_path, attentions, patch_size=16,
                                 image_size=224, save_path=None):
    """
    Visualizes attention rollout map for a ViT model.

    Args:
        image_path: Path to input image.
        attentions: List of attention matrices from ViT forward pass.
        patch_size: Size of each patch in ViT.
        image_size: Input image size.
        save_path: Optional path to save visualization.
    """
    raw_image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    raw_np = np.array(raw_image)

    # Compute rollout
    mask = attention_rollout(attentions)

    # Reshape to spatial grid
    grid_size = image_size // patch_size
    mask = mask.reshape(grid_size, grid_size)
    mask = cv2.resize(mask, (image_size, image_size))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(raw_np)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="viridis")
    axes[1].set_title("Attention Rollout", fontsize=12)
    axes[1].axis("off")

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    blended = (0.6 * raw_np + 0.4 * heatmap_colored).astype(np.uint8)

    axes[2].imshow(blended)
    axes[2].set_title("Attention Overlay", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Attention rollout saved: {save_path}")

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, default="custom_cnn",
                        choices=["custom_cnn", "resnet50", "efficientnet_b3", "mobilenetv2"])
    parser.add_argument("--num_classes", type=int, default=38)
    parser.add_argument("--save_dir", type=str, default="results/gradcam")
    args = parser.parse_args()

    from src.data_loader import test_transform
    from src.utils import get_device

    device = get_device()

    # Build model
    if args.model_type == "custom_cnn":
        from models.custom_cnn import CustomCNN
        model = CustomCNN(args.num_classes)
        target_layer = model.features[-3]  # Last conv block
    elif args.model_type == "resnet50":
        from src.train_resnet50 import build_resnet50
        model = build_resnet50(args.num_classes, freeze_backbone=False)
        target_layer = model.layer4[-1].conv2
    elif args.model_type == "efficientnet_b3":
        from src.train_efficientnet import build_efficientnet_b3
        model = build_efficientnet_b3(args.num_classes, freeze_backbone=False)
        target_layer = model.features[-1]
    elif args.model_type == "mobilenetv2":
        from src.train_mobilenet import build_mobilenetv2
        model = build_mobilenetv2(args.num_classes, freeze_backbone=False)
        target_layer = model.features[-1]

    # Load weights
    checkpoint = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)

    # Generate Grad-CAM
    save_path = os.path.join(args.save_dir, f"gradcam_{args.model_type}.png")
    visualize_gradcam(
        args.image, model, target_layer, test_transform, device,
        save_path=save_path,
    )
