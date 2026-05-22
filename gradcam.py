import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from model import CustomCNN
from torchvision.models import mobilenet_v2
import torch.nn as nn
import pandas as pd
from dataset import EgocartDataset, test_transform
from torch.utils.data import DataLoader
import os
import argparse


# Transform without normalization for visualization
vis_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def load_model(model_name, checkpoint_path, device, num_classes=16):
    """Load model and return it with the appropriate target layer for Grad-CAM."""
    if model_name == "CustomCNN":
        model = CustomCNN(num_classes=num_classes)
        # Last Conv2d block before AdaptiveAvgPool
        target_layer = [model.features[-5]]  # last Conv2d in Sequential
    elif model_name == "MobileNetv2":
        model = mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        # Last convolutional layer of MobileNetV2
        target_layer = [model.features[-1]]

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, target_layer


def find_interesting_samples(model, dataset, device, num_correct=3, num_wrong=3):
    """Find correctly and incorrectly classified samples."""
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    correct_samples = []
    wrong_samples = []

    with torch.no_grad():
        for idx, (image, label) in enumerate(loader):
            image = image.to(device)
            output = model(image)
            _, predicted = torch.max(output, dim=1)

            pred = predicted.item()
            true = label.item()

            if pred == true and len(correct_samples) < num_correct:
                correct_samples.append((idx, true, pred))
            elif pred != true and len(wrong_samples) < num_wrong:
                wrong_samples.append((idx, true, pred))

            if len(correct_samples) >= num_correct and len(wrong_samples) >= num_wrong:
                break

    return correct_samples, wrong_samples


def get_original_image(dataset, idx):
    """Load the original image (without normalization) for visualization."""
    rgb_path = dataset.rgb_paths[idx]
    img = Image.open(rgb_path).convert("RGB")
    img_tensor = vis_transform(img)
    # Convert to numpy HWC format in [0, 1]
    img_np = img_tensor.permute(1, 2, 0).numpy()
    return img_np


def generate_gradcam(model, target_layers, input_tensor, device):
    """Generate Grad-CAM heatmap for an input tensor."""
    input_tensor = input_tensor.unsqueeze(0).to(device)

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    return grayscale_cam


def plot_gradcam_grid(model, target_layers, dataset, samples, title, save_path, device):
    """Plot a grid of original images with Grad-CAM overlays."""
    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))

    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (idx, true_label, pred_label) in enumerate(samples):
        # Original image for visualization
        img_np = get_original_image(dataset, idx)

        # Normalized image for model input
        normalized_img = dataset[idx][0]

        # Generate Grad-CAM
        grayscale_cam = generate_gradcam(model, target_layers, normalized_img, device)

        # Overlay
        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        # Plot original
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"True: {true_label + 1} | Pred: {pred_label + 1}", fontsize=11)
        axes[i, 0].axis("off")

        # Plot Grad-CAM
        axes[i, 1].imshow(cam_image)
        axes[i, 1].set_title("Grad-CAM", fontsize=11)
        axes[i, 1].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["CustomCNN", "MobileNetv2"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default="../egocart/")
    parser.add_argument("--num_correct", type=int, default=3)
    parser.add_argument("--num_wrong", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="gradcam_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    test_path = args.dataset_root + "test_set/"
    test_rgb_path = test_path + "test_RGB/"
    column_names = ["rgb_image_filename", "depth_image_filename", "x", "y", "u", "v", "c"]
    test_df = pd.read_csv(test_path + "test_set.txt", sep=r"\s+", names=column_names)
    test_dataset = EgocartDataset(test_df, test_rgb_path, transform=test_transform)

    # Load model
    model, target_layers = load_model(args.model, args.checkpoint, device)
    print(f"Model: {args.model}, Checkpoint: {args.checkpoint}")

    # Find interesting samples
    print("Finding interesting samples...")
    correct, wrong = find_interesting_samples(
        model, test_dataset, device,
        num_correct=args.num_correct,
        num_wrong=args.num_wrong
    )

    print(f"Found {len(correct)} correct, {len(wrong)} wrong samples")

    # Generate Grad-CAM for correct predictions
    if correct:
        plot_gradcam_grid(
            model, target_layers, test_dataset, correct,
            title=f"{args.model} — Predizioni corrette",
            save_path=os.path.join(args.output_dir, f"gradcam_correct_{args.model.lower()}.png"),
            device=device
        )

    # Generate Grad-CAM for wrong predictions
    if wrong:
        plot_gradcam_grid(
            model, target_layers, test_dataset, wrong,
            title=f"{args.model} — Predizioni errate",
            save_path=os.path.join(args.output_dir, f"gradcam_wrong_{args.model.lower()}.png"),
            device=device
        )