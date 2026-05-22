import json
import time

import torch
from torch import nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from model import CustomCNN
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import pandas as pd
from torch.utils.data import DataLoader
from dataset import EgocartDataset, test_transform, train_transform
from eval import evaluate
import numpy as np


def compute_class_weights(df, num_classes=16):
    """
    Computes inverse frequency class weights to handle class imbalance.
    """
    
    labels = np.array(df.labels).astype(int)

    counts = np.bincount(labels, minlength=num_classes)
    total = labels.shape[0]

    weights = np.zeros(num_classes, dtype=np.float32)
    for i in range(num_classes):
        if counts[i] > 0:
            weights[i] = np.sqrt(float(total) / (num_classes * counts[i]))
        else:
            weights[i] = 0.0

    return torch.from_numpy(weights).float()

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    """Executes one epoch of training. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(data_loader, desc="Training", leave=False)  

    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)  # moltiplica per batch size
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def train(args, paths):
    # Hyperparameters and settings
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size
    num_workers = args.num_workers
    freeze_backbone = args.freeze_backbone
    weighted_loss = args.weighted_loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Running in TRAIN mode")
    
    # Loading the training data in a DataFrame
    print("Loading training data...")
    column_names = ["rgb_image_filename", "depth_image_filename", "x", "y", "u", "v", "c"]
    train_df = pd.read_csv(paths["train_path"] + "train_set.txt", sep="\s+", names=column_names)
    
    # Train/Val split
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, shuffle=True, stratify=train_df["c"])

    # Creating Dataset
    print("Creating datasets and dataloaders...")
    train_dataset = EgocartDataset(train_df, paths["train_rgb_path"], transform=train_transform) 
    val_dataset = EgocartDataset(val_df, paths["train_rgb_path"], transform=test_transform) 

    # Creating Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Model initialization
    print("Initializing model...")
    if args.model == "CustomCNN":
        model = CustomCNN(num_classes=16).to(device)
    elif args.model == "MobileNetv2":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 16) # Substituting the classifier layer
        model.to(device)

    # Loss and optimizer definition
    if weighted_loss:
        class_weights = compute_class_weights(train_dataset).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else: 
        criterion = torch.nn.CrossEntropyLoss()
    if freeze_backbone and args.model == "MobileNetv2":
        print("Freezing backbone layers...")
        for param in model.features.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print("Starting training...")
    best_accuracy = 0.0
    best_epoch = 0
    history = {
        "epochs": [],
        "config": {
            "model": args.model,
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "freeze_backbone": freeze_backbone,
            "weighted_loss": weighted_loss,
            "num_params": sum(p.numel() for p in model.parameters()),
            "device": str(device),
            "peak_gpu_memory_mb": None,
            "iterations_per_epoch": len(train_dataset)//batch_size,
            "mean_epoch_time_sec": None,
            "throughput_samples_per_sec": None,
            "mean_time_per_iteration_sec": None
        },
        "best": {
            "epoch": None,
            "val_acc": None
        }
    }
    torch.cuda.reset_peak_memory_stats() # Reset GPU memory stats at the beginning of training

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        epoch_time = time.time() - start
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        print("Train Loss: {:.4f} | Train Acc: {:.4f} | Val Loss: {:.4f} | Val Acc: {:.4f}".format(
            train_loss, train_acc, val_loss, val_acc
        ))

        # Saving best model
        if val_acc > best_accuracy:
            print("New best model found, saving checkpoint...")
            best_accuracy = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"best_model_{args.model.lower()}.pth")

        # Saving history for analysis
        history["epochs"].append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time": epoch_time
        })

    epoch_times = [e["epoch_time"] for e in history["epochs"]]
    mean_epoch_time = sum(epoch_times) / len(epoch_times)
    iterations_per_epoch = len(train_dataset) // batch_size

    history["config"]["mean_epoch_time_sec"] = mean_epoch_time
    history["config"]["throughput_samples_per_sec"] = len(train_dataset) / mean_epoch_time
    history["config"]["mean_time_per_iteration_sec"] = mean_epoch_time / iterations_per_epoch
    history["best"]["epoch"] = best_epoch
    history["best"]["val_acc"] = best_accuracy
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # in MB
    history["config"]["peak_gpu_memory_mb"] = peak_memory
    print("Training completed.")


    print(f"SUMMARY\n mode: {args.mode}\n model: {args.model}\n freeze_backbone: {freeze_backbone}\n\
            epochs: {epochs}\n learning_rate: {lr}\n batch_size: {batch_size}\n num_workers: {num_workers}\n best_val_accuracy: {best_accuracy:.4f}\n\
            device: {device}\n peak_gpu_memory_mb: {peak_memory:.2f}\n")
    
    with open(f"train_{args.model.lower()}.json", "w") as f:
        json.dump(history, f, indent=2)