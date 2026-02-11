import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models import FakeArtEfficientNet, ViTBaseline


def reshape_transform(tensor, height=8, width=8):
    """Reshape ViT token outputs to a spatial feature map for Grad-CAM."""
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def visualize_gradcam(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if args.model == "efficientnet":
        model = FakeArtEfficientNet(pretrained=False).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        target_layers = [model.backbone.features[-1]]
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        display_transform = T.Compose([T.Resize(256), T.CenterCrop(224)])
        cam_kwargs = {}
    elif args.model == "vit":
        model = ViTBaseline(img_size=128).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        target_layers = [model.vit.blocks[-1].norm1]
        transform = T.Compose([
            T.Resize(144),
            T.CenterCrop(128),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        display_transform = T.Compose([T.Resize(144), T.CenterCrop(128)])
        cam_kwargs = {"reshape_transform": reshape_transform}
    else:
        raise ValueError("Grad-CAM supported for 'efficientnet' and 'vit' only")

    model.eval()

    # Prepare image
    orig_img = Image.open(args.image).convert("RGB")
    input_tensor = transform(orig_img).unsqueeze(0).to(device)
    display_img = display_transform(orig_img)
    hwc_img = np.array(display_img).astype(np.float32) / 255.0

    # Generate Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers, **cam_kwargs)
    targets = [ClassifierOutputTarget(args.target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Visualize
    overlay = show_cam_on_image(hwc_img, grayscale_cam, use_rgb=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(overlay)
    ax1.set_title("Grad-CAM Overlay")
    ax1.axis("off")
    ax2.imshow(grayscale_cam, cmap="jet")
    ax2.set_title("Raw Heatmap")
    ax2.axis("off")
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM visualization")
    parser.add_argument("--model", type=str, required=True,
                        choices=["efficientnet", "vit"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--target-class", type=int, default=0,
                        help="Target class for Grad-CAM (0=AI, 1=Real)")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save output image (displays if not set)")
    args = parser.parse_args()
    visualize_gradcam(args)
