# Smart Agriculture using Deep Learning

> AI-powered plant disease detection system using CNNs, GANs, Diffusion Models, and Vision Transformers.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [API Deployment](#api-deployment)
- [Docker](#docker)
- [Results](#results)
- [Report Structure](#report-structure)

---

## 🌱 Overview

This project implements a comprehensive deep learning pipeline for automated plant disease detection from leaf images. It covers:

1. **CNN Classifiers** — Custom CNN, ResNet50, EfficientNet-B3, MobileNetV2
2. **Generative Models** — DCGAN, Conditional GAN, DDPM for data augmentation
3. **Vision Transformers** — ViT, DeiT, Swin Transformer, Hybrid CNN-ViT
4. **Deployment** — FastAPI REST API with ensemble predictions, Dockerized

---

## 📁 Project Structure

```
Smart-Agriculture-DL/
│
├── api/
│   └── main.py                 # FastAPI backend
│
├── datasets/
│   ├── PlantVillage/           # PlantVillage dataset (ImageFolder)
│   └── PlantDoc/               # PlantDoc dataset (ImageFolder)
│
├── src/
│   ├── data_loader.py          # Dataset loading & augmentation
│   ├── utils.py                # Training utilities & helpers
│   ├── evaluate.py             # Evaluation metrics & plots
│   ├── gradcam.py              # Grad-CAM & attention visualization
│   ├── ensemble.py             # Ensemble prediction system
│   ├── train_custom_cnn.py     # Custom CNN training
│   ├── train_resnet50.py       # ResNet50 transfer learning
│   ├── train_efficientnet.py   # EfficientNet-B3 training
│   ├── train_mobilenet.py      # MobileNetV2 training
│   ├── train_dcgan.py          # DCGAN training
│   ├── train_cgan.py           # Conditional GAN training
│   ├── train_ddpm.py           # DDPM training (U-Net backbone)
│   └── train_vit.py            # ViT / DeiT / Swin / Hybrid training
│
├── models/
│   ├── custom_cnn.py           # Custom CNN architecture
│   └── checkpoints/            # Saved model weights
│
├── results/
│   ├── confusion_matrix/       # Confusion matrix plots
│   ├── gradcam/                # Grad-CAM visualizations
│   ├── dcgan/                  # DCGAN generated samples
│   ├── cgan/                   # Conditional GAN samples
│   ├── ddpm/                   # DDPM generated samples
│   └── logs/                   # Training logs & curves
│
├── notebooks/                  # Jupyter notebooks
├── report/                     # Final report documents
│
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker image configuration
├── docker-compose.yml          # Docker Compose config
├── train.sh                    # Master training script
└── README.md                   # This file
```

---

## 📊 Dataset

| Dataset | Images | Classes | Description |
|---------|--------|---------|-------------|
| **PlantVillage** | 54,305 | 38 | Lab-condition leaf images |
| **PlantDoc** | 2,598 | 27 | Real-world field images |

### Setup

```bash
# Place datasets in ImageFolder format:
datasets/PlantVillage/train/<class_name>/<image>.jpg
datasets/PlantVillage/val/<class_name>/<image>.jpg
```

---

## 🧠 Models

### Task A — CNN Classifiers

| Model | Type | Params | Description |
|-------|------|--------|-------------|
| Custom CNN | Scratch | ~2.5M | 5-block CNN with BatchNorm |
| ResNet50 | Transfer | ~25.6M | Fine-tuned layer4 + classifier |
| EfficientNet-B3 | Transfer | ~12.2M | Fine-tuned last 2 blocks |
| MobileNetV2 | Transfer | ~3.4M | Lightweight, edge-ready |

### Task B — Generative Models

| Model | Type | Description |
|-------|------|-------------|
| DCGAN | GAN | 64×64 unconditional generation |
| Conditional GAN | cGAN | Class-conditional disease synthesis |
| DDPM | Diffusion | U-Net denoising diffusion model |

### Task C — Vision Transformers

| Model | Type | Description |
|-------|------|-------------|
| ViT-B/16 | Transformer | Patch-based vision transformer |
| DeiT-Small | Transformer | Data-efficient ViT with distillation |
| Swin-Tiny | Transformer | Shifted window self-attention |
| Hybrid CNN-ViT | Hybrid | CNN features → Transformer encoder |

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Smart-Agriculture-DL.git
cd Smart-Agriculture-DL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🏋️ Training

### Train All Models

```bash
bash train.sh all
```

### Train by Phase

```bash
bash train.sh cnn    # Phase 1: CNN models
bash train.sh gan    # Phase 2: Generative models
bash train.sh vit    # Phase 3: Vision Transformers
bash train.sh eval   # Phase 4: Evaluation
```

### Train Individual Models

```bash
# CNN Models
python src/train_custom_cnn.py --epochs 30
python src/train_resnet50.py --epochs 25
python src/train_efficientnet.py --epochs 25
python src/train_mobilenet.py --epochs 25

# Generative Models
python src/train_dcgan.py --epochs 100
python src/train_cgan.py --epochs 100
python src/train_ddpm.py --epochs 200

# Vision Transformers
python src/train_vit.py --model vit --epochs 25
python src/train_vit.py --model deit --epochs 25
python src/train_vit.py --model swin --epochs 25
python src/train_vit.py --model hybrid --epochs 25
```

---

## 📈 Evaluation

```bash
python src/evaluate.py \
    --model_type resnet50 \
    --model_path models/checkpoints/resnet50_best.pt \
    --val_dir datasets/PlantVillage/val \
    --train_dir datasets/PlantVillage/train
```

### Grad-CAM Visualization

```bash
python src/gradcam.py \
    --image path/to/leaf.jpg \
    --model_path models/checkpoints/resnet50_best.pt \
    --model_type resnet50 \
    --num_classes 38
```

---

## 🚀 API Deployment

### Run Locally

```bash
uvicorn api.main:app --reload
```

Open Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/models` | List available models |
| GET | `/classes` | List disease classes |
| POST | `/predict` | Single model prediction |
| POST | `/predict/ensemble` | Ensemble prediction |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
    -F "file=@leaf_image.jpg" \
    -F "model_name=hybrid"
```

---

## 🐳 Docker

### Build & Run

```bash
docker build -t smart-agriculture .
docker run -p 8000:8000 smart-agriculture
```

### Docker Compose

```bash
docker-compose up -d
```

---

## 📊 Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Custom CNN | 92.4% | 91.8% |
| MobileNetV2 | 96.2% | 95.9% |
| ResNet50 | 97.8% | 97.4% |
| EfficientNet-B3 | 98.5% | 98.2% |
| ViT | 97.6% | — |
| DeiT | 98.1% | — |
| Swin | 99.0% | — |
| Hybrid CNN-ViT | 99.2% | — |
| **Ensemble** | **99.4%** | — |

**Best Result: Ensemble Accuracy = 99.4%**

---

## 📝 Report Structure (12 Chapters)

1. **Introduction** — Smart Agriculture, DL, Computer Vision
2. **Literature Review** — CNN, GAN, DDPM, Vision Transformers
3. **Methodology** — Dataset, Preprocessing, Augmentation
4. **CNN Models** — Custom CNN, ResNet50, EfficientNet, MobileNet
5. **Generative Models** — DCGAN, Conditional GAN, DDPM
6. **Vision Transformers** — ViT, DeiT, Swin, Hybrid CNN-ViT
7. **Experimental Results** — Accuracy, F1, Confusion Matrix, FID
8. **Explainability** — GradCAM, Attention Rollout, Entropy
9. **Deployment** — FastAPI, Ensemble, Docker
10. **Discussion** — Model Comparison
11. **Limitations** — Dataset Bias, GAN Failure Cases
12. **Conclusion & Future Work**

---

## 📄 License

This project is for academic/research purposes.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push and open a Pull Request
