#!/bin/bash
# ============================================================
# Smart Agriculture - Complete Training Pipeline
# ============================================================
# Usage: bash train.sh [phase]
# Phases: all | cnn | gan | vit | eval
# ============================================================

set -e

DATASET_TRAIN="datasets/PlantVillage/train"
DATASET_VAL="datasets/PlantVillage/val"
EPOCHS_CNN=30
EPOCHS_TL=25
EPOCHS_GAN=100
EPOCHS_DDPM=200
EPOCHS_VIT=25

echo "============================================"
echo "  Smart Agriculture - Training Pipeline"
echo "============================================"

PHASE=${1:-all}

# Phase 1: CNN Models
if [[ "$PHASE" == "all" || "$PHASE" == "cnn" ]]; then
    echo ""
    echo ">>> Phase 1: CNN Models"
    echo "--------------------------------------------"

    echo "[1/4] Training Custom CNN..."
    python src/train_custom_cnn.py \
        --train_dir $DATASET_TRAIN \
        --val_dir $DATASET_VAL \
        --epochs $EPOCHS_CNN

    echo "[2/4] Training ResNet50..."
    python src/train_resnet50.py \
        --train_dir $DATASET_TRAIN \
        --val_dir $DATASET_VAL \
        --epochs $EPOCHS_TL

    echo "[3/4] Training EfficientNet-B3..."
    python src/train_efficientnet.py \
        --train_dir $DATASET_TRAIN \
        --val_dir $DATASET_VAL \
        --epochs $EPOCHS_TL

    echo "[4/4] Training MobileNetV2..."
    python src/train_mobilenet.py \
        --train_dir $DATASET_TRAIN \
        --val_dir $DATASET_VAL \
        --epochs $EPOCHS_TL
fi

# Phase 2: Generative Models
if [[ "$PHASE" == "all" || "$PHASE" == "gan" ]]; then
    echo ""
    echo ">>> Phase 2: Generative Models"
    echo "--------------------------------------------"

    echo "[1/3] Training DCGAN..."
    python src/train_dcgan.py \
        --data_dir $DATASET_TRAIN \
        --epochs $EPOCHS_GAN

    echo "[2/3] Training Conditional GAN..."
    python src/train_cgan.py \
        --data_dir $DATASET_TRAIN \
        --epochs $EPOCHS_GAN

    echo "[3/3] Training DDPM..."
    python src/train_ddpm.py \
        --data_dir $DATASET_TRAIN \
        --epochs $EPOCHS_DDPM
fi

# Phase 3: Vision Transformers
if [[ "$PHASE" == "all" || "$PHASE" == "vit" ]]; then
    echo ""
    echo ">>> Phase 3: Vision Transformers"
    echo "--------------------------------------------"

    echo "[1/4] Training ViT..."
    python src/train_vit.py \
        --model vit \
        --train_dir $DATASET_TRAIN \
        --val_dir $DATASET_VAL \
        --epochs $EPOCHS_VIT

    echo "[2/4] Training DeiT..."
    python src/train_vit.py \
        --model deit \
        --train_dir $DATASET_TRAIN \
        --val_dir $DATASET_VAL \
        --epochs $EPOCHS_VIT

    echo "[3/4] Training Swin Transformer..."
    python src/train_vit.py \
        --model swin \
        --train_dir $DATASET_TRAIN \
        --val_dir $DATASET_VAL \
        --epochs $EPOCHS_VIT

    echo "[4/4] Training Hybrid CNN-ViT..."
    python src/train_vit.py \
        --model hybrid \
        --train_dir $DATASET_TRAIN \
        --val_dir $DATASET_VAL \
        --epochs $EPOCHS_VIT
fi

# Phase 4: Evaluation
if [[ "$PHASE" == "all" || "$PHASE" == "eval" ]]; then
    echo ""
    echo ">>> Phase 4: Evaluation"
    echo "--------------------------------------------"

    for MODEL in custom_cnn resnet50 efficientnet_b3 mobilenetv2; do
        echo "Evaluating $MODEL..."
        python src/evaluate.py \
            --model_type $MODEL \
            --model_path "models/checkpoints/${MODEL}_best.pt" \
            --val_dir $DATASET_VAL \
            --train_dir $DATASET_TRAIN
    done
fi

echo ""
echo "============================================"
echo "  Training Pipeline Complete!"
echo "============================================"
echo ""
echo "Results saved to: results/"
echo "Checkpoints saved to: models/checkpoints/"
echo ""
echo "To start API server:"
echo "  uvicorn api.main:app --reload"
echo ""
echo "To build Docker:"
echo "  docker build -t smart-agriculture ."
echo "  docker run -p 8000:8000 smart-agriculture"
