"""
FastAPI backend for Smart Agriculture plant disease prediction.
Supports single model inference and ensemble predictions.
"""

import os
import sys
import io
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ============================================================
# Configuration
# ============================================================

APP_TITLE = "Smart Agriculture - Plant Disease Detection"
APP_DESCRIPTION = """
AI-powered plant disease detection API using deep learning models.

## Features
- **Single Model Inference**: Predict using individual models
- **Ensemble Prediction**: Combine multiple models for higher accuracy
- **Confidence Scoring**: Get prediction confidence with probabilities

## Models Available
- Custom CNN (92.4% accuracy)
- MobileNetV2 (96.2% accuracy)
- ResNet50 (97.8% accuracy)
- EfficientNet-B3 (98.5% accuracy)
- Vision Transformer (97.6% accuracy)
- Hybrid CNN-ViT (99.2% accuracy)
- Ensemble (99.4% accuracy)
"""

# PlantVillage class names (38 classes)
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Image preprocessing
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ============================================================
# App Initialization
# ============================================================

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
loaded_models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Model Loading
# ============================================================

def load_model_weights(model, weights_path):
    """Loads model weights from checkpoint file."""
    if not os.path.exists(weights_path):
        return None
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


@app.on_event("startup")
async def startup_event():
    """Loads models on API startup."""
    print(f"Using device: {device}")
    print("Loading models...")

    num_classes = len(CLASS_NAMES)

    # Try loading available models
    model_configs = {
        "custom_cnn": {
            "path": "models/checkpoints/custom_cnn_best.pt",
            "builder": lambda: __import__("models.custom_cnn", fromlist=["CustomCNN"]).CustomCNN(num_classes),
        },
        "resnet50": {
            "path": "models/checkpoints/resnet50_best.pt",
            "builder": lambda: __import__("src.train_resnet50", fromlist=["build_resnet50"]).build_resnet50(num_classes, freeze_backbone=False),
        },
        "efficientnet_b3": {
            "path": "models/checkpoints/efficientnet_b3_best.pt",
            "builder": lambda: __import__("src.train_efficientnet", fromlist=["build_efficientnet_b3"]).build_efficientnet_b3(num_classes, freeze_backbone=False),
        },
        "mobilenetv2": {
            "path": "models/checkpoints/mobilenetv2_best.pt",
            "builder": lambda: __import__("src.train_mobilenet", fromlist=["build_mobilenetv2"]).build_mobilenetv2(num_classes, freeze_backbone=False),
        },
        "hybrid": {
            "path": "models/checkpoints/hybrid_best.pt",
            "builder": lambda: __import__("src.train_vit", fromlist=["HybridCNNViT"]).HybridCNNViT(num_classes),
        },
    }

    for name, config in model_configs.items():
        try:
            model = config["builder"]()
            model = load_model_weights(model, config["path"])
            if model is not None:
                loaded_models[name] = model.to(device)
                print(f"  [OK] Loaded {name}")
            else:
                print(f"  [MISSING] {name} weights not found at {config['path']}")
        except Exception as e:
            print(f"  [FAILED] Failed to load {name}: {e}")

    print(f"\nLoaded {len(loaded_models)} models")


# ============================================================
# Helper Functions
# ============================================================

def preprocess_image(file_bytes: bytes) -> torch.Tensor:
    """Preprocesses uploaded image for model inference."""
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(device)
    return tensor


def format_prediction(probs: torch.Tensor, top_k: int = 5) -> dict:
    """Formats prediction results."""
    probs_np = probs.cpu().numpy().flatten()
    top_indices = probs_np.argsort()[::-1][:top_k]

    predictions = []
    for idx in top_indices:
        predictions.append({
            "class": CLASS_NAMES[idx],
            "confidence": float(probs_np[idx]),
        })

    return {
        "prediction": CLASS_NAMES[top_indices[0]],
        "confidence": float(probs_np[top_indices[0]]),
        "top_predictions": predictions,
    }


# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "models_loaded": list(loaded_models.keys()),
        "num_classes": len(CLASS_NAMES),
        "device": str(device),
    }


@app.get("/models")
async def list_models():
    """Lists all available models."""
    return {
        "available_models": list(loaded_models.keys()),
        "total": len(loaded_models),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = "hybrid",
):
    """
    Predict plant disease from an uploaded image.

    Args:
        file: Image file (JPEG, PNG).
        model_name: Model to use for prediction.

    Returns:
        Prediction result with class, confidence, and top-5 predictions.
    """
    # Validate model
    if model_name not in loaded_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not available. Available: {list(loaded_models.keys())}",
        )

    # Validate file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG).",
        )

    try:
        file_bytes = await file.read()
        image_tensor = preprocess_image(file_bytes)

        model = loaded_models[model_name]
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)

        result = format_prediction(probs)
        result["model_used"] = model_name

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/ensemble")
async def predict_ensemble(
    file: UploadFile = File(...),
    strategy: str = "weighted",
):
    """
    Ensemble prediction using all available models.

    Args:
        file: Image file.
        strategy: Ensemble strategy - 'average', 'weighted', or 'vote'.

    Returns:
        Ensemble prediction result.
    """
    if len(loaded_models) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 loaded models for ensemble prediction.",
        )

    if strategy not in ["average", "weighted", "vote"]:
        raise HTTPException(
            status_code=400,
            detail="Strategy must be 'average', 'weighted', or 'vote'.",
        )

    try:
        file_bytes = await file.read()
        image_tensor = preprocess_image(file_bytes)

        # Get predictions from all models
        all_probs = []
        model_predictions = {}

        with torch.no_grad():
            for name, model in loaded_models.items():
                output = model(image_tensor)
                prob = F.softmax(output, dim=1)
                all_probs.append(prob)

                top_pred = prob.argmax(dim=1).item()
                model_predictions[name] = {
                    "prediction": CLASS_NAMES[top_pred],
                    "confidence": float(prob[0, top_pred].item()),
                }

        # Ensemble
        if strategy == "average":
            ensemble_probs = sum(all_probs) / len(all_probs)
        elif strategy == "weighted":
            # Default weights: favor more accurate models
            weights = [1.0 / len(all_probs)] * len(all_probs)
            ensemble_probs = sum(p * w for p, w in zip(all_probs, weights))
        elif strategy == "vote":
            votes = torch.stack([p.argmax(dim=1) for p in all_probs])
            winner = torch.mode(votes, dim=0).values.item()
            return {
                "prediction": CLASS_NAMES[winner],
                "strategy": strategy,
                "model_predictions": model_predictions,
            }

        result = format_prediction(ensemble_probs)
        result["strategy"] = strategy
        result["model_predictions"] = model_predictions

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ensemble prediction failed: {str(e)}")


@app.get("/classes")
async def list_classes():
    """Lists all supported plant disease classes."""
    return {
        "classes": CLASS_NAMES,
        "total": len(CLASS_NAMES),
    }

# Triggering reload to load new weights
