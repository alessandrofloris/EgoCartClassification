import json
import time

import torch
from dataset import EgocartDataset, test_transform
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from model import CustomCNN
from torchvision.models import mobilenet_v2
from utils import plot_confusion_matrix
import pandas as pd

def evaluate(model, data_loader, loss_fn, device):
    """Evaluates the model. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_labels

def eval(args, paths):
    # Hyperparameters and settings
    batch_size = args.batch_size
    num_workers = args.num_workers
    checkpoint_path = args.checkpoint or f"best_model_{args.model.lower()}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Running in Test mode")

    # Loading the test data in a DataFrame
    column_names = ["rgb_image_filename", "depth_image_filename", "x", "y", "u", "v", "c"]
    test_df = pd.read_csv(paths["test_path"] + "test_set.txt", sep="\s+", names=column_names)

    # Creating Dataset
    print("Creating dataset and dataloader...")
    test_dataset = EgocartDataset(test_df, paths["test_rgb_path"], transform=test_transform) 

    # Creating Dataloaders
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Model initialization
    print("Initializing model...")
    if args.model == "CustomCNN":
        model = CustomCNN(num_classes=16)
    elif args.model == "MobileNetv2":
        model = mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 16) # Substituting the classifier layer

    model.load_state_dict(torch.load(checkpoint_path)) # loading the best model checkpoint
    model.to(device)

    # Loss definition
    criterion = torch.nn.CrossEntropyLoss()

    # Evaluation 
    print("Evaluating on test set...")
    start = time.time()
    loss, accuracy, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    inference_time = time.time() - start

    report = classification_report(test_labels, test_preds, target_names=[f"Class {i+1}" for i in range(16)], output_dict=True)
    evaluation = {
        "test_loss": loss,
        "test_acc": accuracy,
        "inference_time": inference_time,
        "per_class": report,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "weighted_precision": report["weighted avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "weighted_recall": report["weighted avg"]["recall"]
    }
    
    print("Test Loss: {:.4f} | Test Acc: {:.4f}".format(loss, accuracy))
    print("Inference time: {:.4f} seconds".format(inference_time))
    
    with open(f"evaluation_{args.model.lower()}.json", "w") as f:
        json.dump(evaluation, f, indent=4)

    plot_confusion_matrix(model, test_loader, device)