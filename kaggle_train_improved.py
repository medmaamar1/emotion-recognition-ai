#!/usr/bin/env python3
"""
Improved Kaggle Training Script for Emotion Recognition AI
This script implements class imbalance remediation strategies.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support

# Import configuration and modules
from config import get_config
from scripts.dataset import RAFCEDataset
from models.baseline import FERBaseline
from utils.logger import setup_logger
from utils.losses import FocalLoss
from utils.mixup_cutmix import mixup_data, mixup_criterion
from training.schedulers import CosineAnnealingWarmupScheduler

# Minority classes that need strong augmentation
MINORITY_CLASSES = [4, 8, 12, 13]  # Classes with <3% of data

def get_class_aware_transform(use_augmentation=True):
    """
    Get data transforms with class-specific augmentation.
    
    Args:
        use_augmentation: Whether to use augmentation
    
    Returns:
        train_transform, val_transform, and a function to apply class-aware augmentation
    """
    if use_augmentation:
        # Base augmentation for majority classes
        base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Strong augmentation for minority classes
        strong_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.7),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        strong_transform = base_transform
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return base_transform, strong_transform, val_transform

class ClassAwareDataset:
    """
    Wrapper for dataset that applies class-specific augmentation.
    """
    def __init__(self, base_dataset, base_transform, strong_transform, minority_classes):
        self.base_dataset = base_dataset
        self.base_transform = base_transform
        self.strong_transform = strong_transform
        self.minority_classes = minority_classes
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        label = item['label'].item() if torch.is_tensor(item['label']) else item['label']
        
        # Apply class-specific augmentation
        if label in self.minority_classes:
            # Apply strong augmentation 2x more frequently
            if np.random.random() < 0.7:
                item['image'] = self.strong_transform(item['image'])
            else:
                item['image'] = self.base_transform(item['image'])
        else:
            item['image'] = self.base_transform(item['image'])
        
        return item

def create_data_loaders(config, batch_size=32, use_augmentation=True, use_mixup=False):
    """Create training and validation data loaders."""
    logger = setup_logger(config['log_dir'], 'data_loading')
    
    base_transform, strong_transform, val_transform = get_class_aware_transform(use_augmentation)
    
    # Load datasets
    logger.info("Loading training dataset...")
    base_train_dataset = RAFCEDataset(partition_id=0, transform=None, use_aligned=True)
    
    logger.info("Loading validation dataset...")
    val_dataset = RAFCEDataset(partition_id=2, transform=val_transform, use_aligned=True)
    
    # Wrap training dataset with class-aware augmentation
    train_dataset = ClassAwareDataset(
        base_train_dataset, 
        base_transform, 
        strong_transform, 
        MINORITY_CLASSES
    )
    
    # Calculate class weights for imbalanced data
    targets = [base_train_dataset.emotions[img_id] for img_id in base_train_dataset.image_ids]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[targets]
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create data loaders
    num_workers = config['train_config']['num_workers']
    pin_memory = config['train_config']['pin_memory']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Class distribution: {class_counts}")
    logger.info(f"Minority classes (strong augmentation): {MINORITY_CLASSES}")
    
    return train_loader, val_loader, class_weights

def calculate_per_class_metrics(y_true, y_pred, num_classes=14):
    """
    Calculate per-class metrics (precision, recall, F1).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
    
    Returns:
        Dictionary with per-class and aggregated metrics
    """
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Calculate macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Calculate weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    metrics = {
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'per_class_support': support,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
    }
    
    return metrics

def train_model(config, args):
    """Train the emotion recognition model with improved strategies."""
    # Setup logger
    logger = setup_logger(config['log_dir'], 'training')
    logger.info("=" * 60)
    logger.info("Starting Training with Class Imbalance Remediation")
    logger.info("=" * 60)
    
    # Log configuration
    logger.info(f"Running on Kaggle: {config['is_kaggle']}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Use augmentation: {args.augmentation}")
    logger.info(f"Use Focal Loss: {args.use_focal}")
    logger.info(f"Use MixUp: {args.use_mixup}")
    logger.info(f"Focal gamma: {args.focal_gamma}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, class_weights = create_data_loaders(
        config, 
        batch_size=args.batch_size,
        use_augmentation=args.augmentation,
        use_mixup=args.use_mixup
    )
    
    # Create model
    logger.info(f"Creating {args.model} model...")
    model = FERBaseline(model_type=args.model, num_classes=config['model_config']['num_classes']).to(device)
    
    # Loss function - Focal Loss for class imbalance
    if args.use_focal:
        # Calculate alpha based on inverse class frequency
        targets = [RAFCEDataset(partition_id=0, transform=None, use_aligned=True).emotions[img_id] 
                  for img_id in RAFCEDataset(partition_id=0, transform=None, use_aligned=True).image_ids]
        class_counts = np.bincount(targets)
        alpha = 1.0 / class_counts
        alpha = alpha / alpha.sum()  # Normalize
        
        criterion = FocalLoss(alpha=alpha.tolist(), gamma=args.focal_gamma)
        logger.info(f"Using Focal Loss with gamma={args.focal_gamma}")
        logger.info(f"Alpha weights: {alpha.tolist()}")
    else:
        # Standard CrossEntropyLoss (no class weights since we use WeightedRandomSampler)
        criterion = nn.CrossEntropyLoss()
        logger.info("Using standard CrossEntropyLoss (no class weights)")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingWarmupScheduler(
        optimizer,
        warmup_epochs=5,
        max_epochs=args.epochs
    )
    
    # Training loop
    best_macro_f1 = 0.0
    best_epoch = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    macro_f1s = []
    patience_counter = 0
    patience = 10
    
    logger.info("Starting training loop...")
    logger.info(f"Early stopping patience: {patience} epochs")
    logger.info(f"Monitoring metric: Macro-F1 (not accuracy)")
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # MixUp augmentation
            if args.use_mixup and np.random.random() < 0.5:
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.4)
                optimizer.zero_grad()
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
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
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for batch in pbar:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': val_loss/len(val_loader), 'acc': 100 * val_correct / val_total})
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Calculate per-class metrics
        metrics = calculate_per_class_metrics(val_labels, val_preds, num_classes=14)
        macro_f1 = metrics['macro_f1']
        macro_f1s.append(macro_f1)
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{args.epochs}: "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                   f"Macro-F1: {macro_f1:.4f}")
        
        # Log per-class F1 for minority classes
        minority_f1 = [metrics['per_class_f1'][i] for i in MINORITY_CLASSES]
        logger.info(f"Minority Class F1: {[f'{i}:{f:.2f}' for i, f in zip(MINORITY_CLASSES, minority_f1)]}")
        
        # Save best model based on macro-F1
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch + 1
            patience_counter = 0
            checkpoint_path = os.path.join(config['checkpoint_dir'], f"{args.model}_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'macro_f1': macro_f1,
                'per_class_metrics': metrics,
                'config': config
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path} (Macro-F1: {macro_f1:.4f})")
        else:
            patience_counter += 1
            logger.info(f"Patience counter: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            logger.info(f"Best Macro-F1: {best_macro_f1:.4f} at epoch {best_epoch}")
            break
        
        # Step scheduler
        scheduler.step()
    
    # Save final model
    final_checkpoint_path = os.path.join(config['checkpoint_dir'], f"{args.model}_final.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'train_loss': train_loss,
        'macro_f1': macro_f1,
        'per_class_metrics': metrics,
        'config': config
    }, final_checkpoint_path)
    logger.info(f"Saved final model to {final_checkpoint_path}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'macro_f1s': macro_f1s,
        'best_macro_f1': best_macro_f1,
        'best_epoch': best_epoch,
        'args': vars(args)
    }
    
    history_path = os.path.join(config['output_dir'], f"{args.model}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    logger.info(f"Saved training history to {history_path}")
    
    # Save per-class metrics report
    report = classification_report(val_labels, val_preds, output_dict=True, zero_division=0)
    report_path = os.path.join(config['output_dir'], f"{args.model}_classification_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    logger.info(f"Saved classification report to {report_path}")
    
    logger.info("=" * 60)
    logger.info("Training Completed")
    logger.info(f"Best Macro-F1: {best_macro_f1:.4f} at epoch {best_epoch}")
    logger.info(f"Best Validation Accuracy: {val_accs[best_epoch-1]:.2f}%")
    logger.info("=" * 60)
    
    return model, history

def main():
    """Main function for Kaggle training."""
    parser = argparse.ArgumentParser(description='Train emotion recognition model on Kaggle with class imbalance fixes')
    parser.add_argument('--model', type=str, default='resnet50', 
                       choices=['resnet50', 'efficientnet_b0', 'vit_base_patch16_224'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--augmentation', action='store_true', default=True,
                       help='Use data augmentation')
    parser.add_argument('--no_augmentation', action='store_false', dest='augmentation',
                       help='Disable data augmentation')
    parser.add_argument('--use_focal', action='store_true', default=True,
                       help='Use Focal Loss for class imbalance')
    parser.add_argument('--no_focal', action='store_false', dest='use_focal',
                       help='Disable Focal Loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Gamma parameter for Focal Loss')
    parser.add_argument('--use_mixup', action='store_true', default=True,
                       help='Use MixUp augmentation')
    parser.add_argument('--no_mixup', action='store_false', dest='use_mixup',
                       help='Disable MixUp augmentation')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config()
    
    # Print configuration
    print("=" * 60)
    print("Kaggle Training Configuration (Improved)")
    print("=" * 60)
    print(f"Running on Kaggle: {config['is_kaggle']}")
    print(f"Data Root (Aligned): {config['data_root_aligned']}")
    print(f"Output Directory: {config['output_dir']}")
    print(f"Checkpoint Directory: {config['checkpoint_dir']}")
    print(f"Log Directory: {config['log_dir']}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Augmentation: {args.augmentation}")
    print(f"Focal Loss: {args.use_focal}")
    print(f"Focal Gamma: {args.focal_gamma}")
    print(f"MixUp: {args.use_mixup}")
    print("=" * 60)
    print()
    
    # Train model
    model, history = train_model(config, args)
    
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Best Macro-F1: {history['best_macro_f1']:.4f}")
    print(f"Best Epoch: {history['best_epoch']}")
    print(f"Final Validation Accuracy: {history['val_accs'][-1]:.2f}%")
    print(f"Final Training Accuracy: {history['train_accs'][-1]:.2f}%")
    print(f"Train-Val Gap: {history['train_accs'][-1] - history['val_accs'][-1]:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
