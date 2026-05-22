import argparse 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch

def parse_args():
    parser = argparse.ArgumentParser(prog="EgocartDeepLearning", description="Egocart script for training and evaluating")
    subparsers = parser.add_subparsers(dest="mode", description="Mode of operation")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--model", choices=["CustomCNN", "MobileNetv2"], default="CustomCNN", help="Model architecture to use for training")
    train_parser.add_argument("--freeze_backbone", action="store_true", help="Whether to freeze the backbone layers during training (only applicable for transfer learning)")
    train_parser.add_argument("--weighted_loss", action="store_true", help="Whether to use weighted loss during training")
    train_parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training")
    train_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    train_parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")

    eval_parser = subparsers.add_parser("eval", help="Evaluate the model on the test set")
    eval_parser.add_argument("--model", choices=["CustomCNN", "MobileNetv2"], default="CustomCNN", help="Model architecture to use for evaluation")
    eval_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    eval_parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    eval_parser.add_argument("--checkpoint", type=str, help="Path to the model checkpoint for evaluation")
    
    return parser.parse_args()

def save_experiment_results(args, best_accuracy, device):
    with open("experiment_results.txt", "w") as f:
        f.write(f"SUMMARY\n mode: {args.mode}\n model: {args.model}\n freeze_backbone: {args.freeze_backbone}\n\
          epochs: {args.epochs}\n learning_rate: {args.lr}\n batch_size: {args.batch_size}\n num_workers: {args.num_workers}\n best_val_accuracy: {best_accuracy:.4f}\n\
            device: {device}\n")

    
def plot_confusion_matrix(model, data_loader, device, num_classes=16):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(1, num_classes + 1),
                yticklabels=range(1, num_classes + 1))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()