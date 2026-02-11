import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm


class FakeArtResNet(nn.Module):
    """ResNet-18 baseline for binary classification of real vs AI-generated art."""

    def __init__(self, pretrained=False):
        super().__init__()
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,
            progress=False,
        )
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.model(x)


class FakeArtEfficientNet(nn.Module):
    """EfficientNet-B0 for binary classification of real vs AI-generated art.

    Uses compound scaling and transfer learning from ImageNet for strong
    performance with fewer parameters than ResNet variants.
    """

    def __init__(self, pretrained=True, dropout_p=0.5):
        super().__init__()
        weights = (
            models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = models.efficientnet_b0(weights=weights, progress=False)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        return self.backbone(x)


class ViTBaseline(nn.Module):
    """Vision Transformer (ViT-B/16) for binary classification.

    Adapts the transformer architecture to process 128x128 images by
    overriding the default patch embedding size.
    """

    def __init__(self, img_size=128):
        super().__init__()
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            img_size=img_size,
        )
        in_features = self.vit.head.in_features
        self.vit.head = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.vit(x)


class HybridCNNViT(nn.Module):
    """Hybrid CNN-ViT model combining local and global feature extraction.

    Architecture:
        1. EfficientNet-B0 extracts local features (7x7x1280 feature maps)
        2. Learned embedding layer projects CNN features to 128x128x3
        3. ViT-B/16 processes embedded features for global context
        4. Linear classifier produces binary prediction

    This bridges convolutional local feature extraction with transformer-based
    global context modeling for robust AI-generated art detection.
    """

    def __init__(self, cnn, vit, cnn_out_dim=1280, embedding_dim_1=512, embedding_dim_2=256):
        super().__init__()
        self.cnn = cnn
        self.vit = vit
        self.advanced_embedding = nn.Sequential(
            nn.Conv2d(cnn_out_dim, embedding_dim_1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(embedding_dim_1),
            nn.Conv2d(embedding_dim_1, embedding_dim_2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(embedding_dim_2),
            nn.Conv2d(embedding_dim_2, 3, kernel_size=1),
        )
        self.classifier = nn.Linear(self.vit.num_features, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = self.advanced_embedding(x)
        x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        x = self.vit.forward_features(x)
        if isinstance(x, tuple):
            x = x[0]
        x = x[:, 0, :]
        x = self.classifier(x)
        return x

    @staticmethod
    def from_pretrained(efficientnet_path, vit_path, device="cpu"):
        """Build a hybrid model from pretrained EfficientNet and ViT checkpoints.

        Args:
            efficientnet_path: Path to saved EfficientNet state dict.
            vit_path: Path to saved ViT state dict.
            device: Device to load the model onto.

        Returns:
            HybridCNNViT model with loaded weights (embedding layer untrained).
        """
        # Load fine-tuned EfficientNet and strip its classifier
        eff_model = FakeArtEfficientNet(pretrained=False)
        eff_model.load_state_dict(torch.load(efficientnet_path, map_location=device))
        base_cnn = eff_model.backbone
        base_cnn.fc = nn.Identity()
        cnn = nn.Sequential(*list(base_cnn.children())[:-2])

        # Load fine-tuned ViT and strip its classification head
        vit_model = ViTBaseline()
        vit_model.load_state_dict(torch.load(vit_path, map_location=device))
        vit_model.vit.head = nn.Identity()
        vit = vit_model.vit

        return HybridCNNViT(cnn=cnn, vit=vit).to(device)
