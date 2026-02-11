# BridgeNet | Hybrid Vision Model for AI Art Detection

A hybrid deep learning architecture that **bridges** Convolutional Neural Networks (EfficientNet-B0) with Vision Transformers (ViT-B/16) to detect diffusion-based AI-generated artworks, achieving **99.5% accuracy** on a 30,000-image test set.

## Overview

The rise of diffusion models like Stable Diffusion and DALL-E has made AI-generated artworks nearly indistinguishable from human-made pieces, raising concerns around authenticity, copyright, and artistic attribution. This project explores deep learning approaches for automated detection, culminating in a novel **hybrid CNN-ViT architecture** that bridges local feature extraction with global context modeling.

## Architecture

Four model architectures are implemented and compared:

### 1. CNN Baselines (ResNet-18 & EfficientNet-B0)

Standard convolutional networks fine-tuned for binary classification. EfficientNet-B0 leverages compound scaling for strong performance with fewer parameters.

### 2. Vision Transformer (ViT-B/16)

Processes images as sequences of 16x16 patches, using self-attention to capture global dependencies across the entire image.

### 3. Hybrid CNN-ViT

The core contribution — a pipeline that chains CNN local feature extraction with ViT global context modeling:

```
Input Image (128x128x3)
       |
  EfficientNet-B0 (pretrained, classifier removed)
       |
  Feature Maps (7x7x1280)
       |
  Embedding Layer (Conv2d blocks + BatchNorm + ReLU)
       |
  Interpolate to 128x128x3
       |
  ViT-B/16 Transformer Encoder
       |
  [CLS] Token -> Linear -> Sigmoid
       |
  Binary Prediction (Real / AI-Generated)
```

The embedding layer learns to translate CNN feature maps into the patch embedding format expected by the ViT, enabling end-to-end training of the full pipeline.

### 4. Ensemble

Averages logits from independently trained EfficientNet and ViT models before applying sigmoid activation.

## Results

Evaluated on 30,000 unseen test images (20,000 AI-generated + 10,000 real):

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| ResNet-18 | 0.8766 | 0.7365 | 0.9805 | 0.8412 |
| **EfficientNet-B0** | **0.9955** | **0.9997** | 0.9976 | **0.9984** |
| ViT-B/16 | 0.9979 | 0.9920 | 0.9949 | 0.9966 |
| Hybrid CNN-ViT | 0.9892 | 0.9865 | 0.9810 | 0.9838 |
| Ensemble | 0.9946 | 0.9958 | 0.9952 | 0.9968 |

**Key findings:**
- EfficientNet-B0 and ViT individually achieve near-perfect accuracy (>99.5%)
- The **hybrid model** provides the most balanced precision/recall trade-off, important when false positives and false negatives carry different costs
- The **ensemble** nearly matches the best single-model performance while improving robustness through model diversity

## Dataset

Uses the [AI-ArtBench](https://www.kaggle.com/datasets/ravidussilva/real-ai-art) dataset from Kaggle:
- **155,015 training images** (105,015 AI-generated via Latent/Standard Diffusion + 50,000 real)
- **30,000 test images** (20,000 AI-generated + 10,000 real)
- **10 art styles** (Art Nouveau, Baroque, Expressionism, etc.)
- Training set is balanced to 50/50 via undersampling, then split 80/20 for train/validation

## Project Structure

```
bridgenet/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── models.py              # All model architectures
    ├── dataset.py             # Dataset class and data loaders
    ├── train.py               # Training script with CLI
    ├── evaluate.py            # Evaluation on test set
    ├── gradcam.py             # Grad-CAM interpretability visualizations
    └── tune_hyperparams.py    # Ray Tune hyperparameter optimization
```

## Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/bridgenet.git
cd bridgenet

# Install dependencies
pip install -r requirements.txt

# Download the dataset from Kaggle
pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('ravidussilva/real-ai-art')"
```

## Usage

### Training

```bash
# Train EfficientNet-B0
python src/train.py --model efficientnet --train-dir data/train --test-dir data/test \
    --epochs 5 --batch-size 64 --lr 4.2e-4 --save-path checkpoints/efficientnet.pth

# Train ViT-B/16
python src/train.py --model vit --train-dir data/train --test-dir data/test \
    --epochs 4 --batch-size 64 --lr 1.07e-5 --clip-grad --save-path checkpoints/vit.pth

# Train Hybrid CNN-ViT (requires pretrained CNN and ViT checkpoints)
python src/train.py --model hybrid --train-dir data/train --test-dir data/test \
    --epochs 4 --batch-size 64 --lr 1.07e-5 --clip-grad \
    --cnn-checkpoint checkpoints/efficientnet.pth \
    --vit-checkpoint checkpoints/vit.pth \
    --save-path checkpoints/hybrid.pth
```

### Hyperparameter Tuning

```bash
python src/tune_hyperparams.py --model efficientnet \
    --train-dir data/train --test-dir data/test --num-samples 12
```

### Evaluation

```bash
# Evaluate a single model
python src/evaluate.py --model efficientnet --checkpoint checkpoints/efficientnet.pth \
    --train-dir data/train --test-dir data/test

# Evaluate ensemble
python src/evaluate.py --model ensemble --train-dir data/train --test-dir data/test \
    --cnn-checkpoint checkpoints/efficientnet.pth \
    --vit-checkpoint checkpoints/vit.pth
```

### Grad-CAM Visualization

```bash
# EfficientNet Grad-CAM
python src/gradcam.py --model efficientnet --checkpoint checkpoints/efficientnet.pth \
    --image path/to/image.jpg --save output/gradcam_eff.png

# ViT Grad-CAM
python src/gradcam.py --model vit --checkpoint checkpoints/vit.pth \
    --image path/to/image.jpg --target-class 0 --save output/gradcam_vit.png
```

## Training Details

- **Loss function**: BCEWithLogitsLoss (binary cross-entropy with logits)
- **Optimizer**: AdamW with decoupled weight decay
- **Regularization**: Dropout (p=0.5), weight decay, gradient clipping (max norm 1.0)
- **Hyperparameter tuning**: Ray Tune with ASHA early stopping scheduler
- **Input resolution**: 128x128 (downsampled from 224x224 for compute efficiency)
- **Preprocessing**: ImageNet normalization, random horizontal flip augmentation
- **Hardware**: NVIDIA Tesla K80 GPU (Google Colab)

## Technical Highlights

- **Transfer learning** from ImageNet-pretrained weights for both CNN and ViT backbones
- **Balanced training** via undersampling to handle class imbalance (2:1 AI-to-real ratio)
- **Automated hyperparameter search** using Ray Tune with ASHA scheduling across learning rate, batch size, and weight decay
- **Grad-CAM interpretability** for both CNN and ViT architectures, with custom reshape transform for ViT attention maps
- **Compute-efficient training** through mini-batch capping and input downsampling

## References

1. Silva et al. "ArtBrain: An Explainable End-to-End Toolkit for Classification and Attribution of AI-Generated Art and Style" (2024)
2. Bianco et al. "Identifying AI-Generated Art with Deep Learning" — CREAI 2023
3. Velasco et al. "Art Authentication: A Comparative Analysis of CNN Architectures for Detecting AI-Generated and Human-Made Digital Artworks" — ACM 2025
4. Cetinic et al. "Fine-tuning Convolutional Neural Networks for Fine Art Classification" — Expert Systems with Applications (2018)

## Acknowledgements

This project was developed as a group project for the ST311 course. The hybrid CNN-ViT architecture was designed to bridge the complementary strengths of convolutional and attention-based models for robust AI-generated art detection.
