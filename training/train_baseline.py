import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import get_config
from scripts.dataset import RAFCEDataset
from models.baseline import FERBaseline

print("Starting script execution...")

def train_model(model_type='resnet50', epochs=20, batch_size=32, lr=1e-4):
    """Train the baseline model using config-based paths."""
    # Load configuration
    config = get_config()
    print(f"Running on Kaggle: {config['is_kaggle']}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Output directory: {config['output_dir']}")
    print(f"Checkpoint directory: {config['checkpoint_dir']}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets - using config defaults
    train_dataset = RAFCEDataset(partition_id=0, transform=train_transform, use_aligned=True)
    val_dataset = RAFCEDataset(partition_id=2, transform=val_transform, use_aligned=True)

    # Calculate Weights for Imbalance Mitigation
    targets = [train_dataset.emotions[img_id] for img_id in train_dataset.image_ids]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[targets]
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    print("Initializing DataLoaders...")
    num_workers = config['train_config']['num_workers']
    pin_memory = config['train_config']['pin_memory']
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                             num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_memory)

    # Model, Loss, Optimizer
    print(f"Building {model_type} model...")
    model = FERBaseline(model_type=model_type, num_classes=14).to(device)
    
    # Use Weighted CrossEntropy Loss
    print("Calculating loss weights...")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0
    print("Starting training loop...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/len(train_loader), 'acc': 100 * correct / total})

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save to checkpoint directory from config
            checkpoint_path = os.path.join(config['checkpoint_dir'], f"{model_type}_best.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}!")
            
        scheduler.step()

if __name__ == "__main__":
    # To run: python training/train_baseline.py
    # Or on Kaggle: python training/train_baseline.py --epochs 50 --batch_size 32
    import argparse
    
    parser = argparse.ArgumentParser(description='Train baseline emotion recognition model')
    parser.add_argument('--model_type', type=str, default='resnet50', help='Model architecture')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Model Type: {args.model_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 60)
    print()
    
    train_model(model_type=args.model_type, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr)
