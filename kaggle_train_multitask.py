#!/usr/bin/env python3
"""
Multi-Task Training Script for Emotion Recognition with AU Supervision
This script uses Action Unit (AU) labels as auxiliary supervision to improve emotion recognition.
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
from scripts.dataset import RAFCEDataset, AU_LABELS
from models.multitask import FERMultiTask, FERMultiTaskWithAUAttention
from utils.logger import setup_logger
from training.schedulers import CosineAnnealingWarmupScheduler

def get_transforms(use_augmentation=True):
    """Get data transforms for training and validation."""
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_data_loaders(config, batch_size=32, use_augmentation=True):
    """Create training and validation data loaders."""
    logger = setup_logger(config['log_dir'], 'data_loading')
    
    train_transform, val_transform = get_transforms(use_augmentation)
    
    # Load datasets with AU vectors
    logger.info("Loading training dataset with AU labels...")
    train_dataset = RAFCEDataset(
        partition_id=0, 
        transform=train_transform, 
        use_aligned=True,
        return_au_vector=True
    )
    
    logger.info("Loading validation dataset with AU labels...")
    val_dataset = RAFCEDataset(
        partition_id=2, 
        transform=val_transform, 
        use_aligned=True,
        return_au_vector=True
    )
    
    # Calculate class weights for imbalanced data (only for emotion classification)
    targets = [train_dataset.emotions[img_id] for img_id in train_dataset.image_ids]
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
    logger.info(f"Emotion class distribution: {class_counts}")
    logger.info(f"Number of AUs: {len(AU_LABELS)}")
    
    return train_loader, val_loader, class_weights

def calculate_per_class_metrics(y_true, y_pred, num_classes=14):
    """
    Calculate per-class metrics (precision, recall, F1).
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
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

def calculate_au_metrics(au_true, au_pred, num_aus=18):
    """
    Calculate AU prediction metrics (multi-label classification).
    """
    # AU predictions are logits, apply sigmoid to get probabilities
    au_pred_prob = torch.sigmoid(torch.tensor(au_pred))
    au_pred_binary = (au_pred_prob > 0.5).numpy().astype(int)
    
    # Calculate metrics per AU
    au_f1s = []
    au_precisions = []
    au_recalls = []
    
    for i in range(num_aus):
        if np.sum(au_true[:, i]) > 0:  # Only calculate if AU is present in validation set
            precision, recall, f1, _ = precision_recall_fscore_support(
                au_true[:, i], au_pred_binary[:, i], average='binary', zero_division=0
            )
            au_f1s.append(f1)
            au_precisions.append(precision)
            au_recalls.append(recall)
    
    metrics = {
        'au_mean_f1': np.mean(au_f1s) if au_f1s else 0.0,
        'au_mean_precision': np.mean(au_precisions) if au_precisions else 0.0,
        'au_mean_recall': np.mean(au_recalls) if au_recalls else 0.0,
        'num_aus_with_samples': len(au_f1s)
    }
    
    return metrics

def train_model(config, args):
    """Train multi-task emotion recognition model with AU supervision."""
    # Setup logger
    logger = setup_logger(config['log_dir'], 'training')
    logger.info("=" * 60)
    logger.info("Starting Multi-Task Training with AU Supervision")
    logger.info("=" * 60)
    
    # Log configuration
    logger.info(f"Running on Kaggle: {config['is_kaggle']}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Use AU attention: {args.use_au_attention}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"AU loss weight: {args.au_loss_weight}")
    logger.info(f"Use augmentation: {args.augmentation}")
    logger.info(f"Early stopping patience: {args.patience}")
    logger.info(f"Monitor metric: {args.monitor_metric}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, class_weights = create_data_loaders(
        config, 
        batch_size=args.batch_size,
        use_augmentation=args.augmentation
    )
    
    # Create model
    logger.info(f"Creating {args.model} model with multi-task learning...")
    if args.use_au_attention:
        model = FERMultiTaskWithAUAttention(
            backbone=args.model, 
            num_emotions=config['model_config']['num_classes'],
            num_aus=config['model_config']['num_au'],
            pretrained=True,
            dropout=args.dropout_rate
        ).to(device)
    else:
        model = FERMultiTask(
            backbone=args.model, 
            num_emotions=config['model_config']['num_classes'],
            num_aus=config['model_config']['num_au'],
            pretrained=True,
            dropout=args.dropout_rate
        ).to(device)
    
    # Loss functions
    # Emotion classification loss (no class weights since we use WeightedRandomSampler)
    emotion_criterion = nn.CrossEntropyLoss()
    # AU detection loss (binary cross entropy for multi-label classification)
    au_criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingWarmupScheduler(
        optimizer,
        warmup_epochs=5,
        max_epochs=args.epochs
    )
    
    # Training loop
    best_metric = 0.0
    best_epoch = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    macro_f1s = []
    au_f1s = []
    patience_counter = 0
    
    logger.info("Starting training loop...")
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_emotion_loss = 0.0
        running_au_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            images = batch['image'].to(device)
            emotion_labels = batch['label'].to(device)
            au_labels = batch['au_vector'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if args.use_au_attention:
                emotion_logits, au_logits, au_attention = model(images)
            else:
                emotion_logits, au_logits = model(images)
            
            # Calculate losses
            emotion_loss = emotion_criterion(emotion_logits, emotion_labels)
            au_loss = au_criterion(au_logits, au_labels)
            total_loss = emotion_loss + args.au_loss_weight * au_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            running_emotion_loss += emotion_loss.item()
            running_au_loss += au_loss.item()
            _, predicted = torch.max(emotion_logits.data, 1)
            total += emotion_labels.size(0)
            correct += (predicted == emotion_labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss/len(train_loader),
                'emotion_loss': running_emotion_loss/len(train_loader),
                'au_loss': running_au_loss/len(train_loader),
                'acc': 100 * correct / total
            })
        
        train_loss = running_loss / len(train_loader)
        train_emotion_loss = running_emotion_loss / len(train_loader)
        train_au_loss = running_au_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_emotion_loss = 0.0
        val_au_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        val_au_preds = []
        val_au_true = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for batch in pbar:
                images = batch['image'].to(device)
                emotion_labels = batch['label'].to(device)
                au_labels = batch['au_vector'].to(device)
                
                # Forward pass
                if args.use_au_attention:
                    emotion_logits, au_logits, au_attention = model(images)
                else:
                    emotion_logits, au_logits = model(images)
                
                # Calculate losses
                emotion_loss = emotion_criterion(emotion_logits, emotion_labels)
                au_loss = au_criterion(au_logits, au_labels)
                total_loss = emotion_loss + args.au_loss_weight * au_loss
                
                val_loss += total_loss.item()
                val_emotion_loss += emotion_loss.item()
                val_au_loss += au_loss.item()
                _, predicted = torch.max(emotion_logits.data, 1)
                val_total += emotion_labels.size(0)
                val_correct += (predicted == emotion_labels).sum().item()
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(emotion_labels.cpu().numpy())
                val_au_preds.extend(au_logits.cpu().numpy())
                val_au_true.extend(au_labels.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': val_loss/len(val_loader),
                    'emotion_loss': val_emotion_loss/len(val_loader),
                    'au_loss': val_au_loss/len(val_loader),
                    'acc': 100 * val_correct / val_total
                })
        
        val_loss = val_loss / len(val_loader)
        val_emotion_loss = val_emotion_loss / len(val_loader)
        val_au_loss = val_au_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Calculate metrics based on monitor setting
        metrics = calculate_per_class_metrics(val_labels, val_preds, num_classes=14)
        au_metrics = calculate_au_metrics(np.array(val_au_true), np.array(val_au_preds), num_aus=18)
        
        if args.monitor_metric == 'accuracy':
            current_metric = val_acc
            metric_name = 'Val Acc'
        elif args.monitor_metric == 'macro_f1':
            current_metric = metrics['macro_f1']
            metric_name = 'Macro-F1'
        else:  # weighted_f1
            current_metric = metrics['weighted_f1']
            metric_name = 'Weighted-F1'
        
        macro_f1s.append(metrics['macro_f1'])
        au_f1s.append(au_metrics['au_mean_f1'])
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{args.epochs}: "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                   f"Macro-F1: {metrics['macro_f1']:.4f}, "
                   f"AU-F1: {au_metrics['au_mean_f1']:.4f}, "
                   f"{metric_name}: {current_metric:.4f}")
        
        logger.info(f"  Emotion Loss: {val_emotion_loss:.4f}, AU Loss: {val_au_loss:.4f}")
        
        # Log per-class F1 for minority classes
        minority_classes = [4, 8, 12, 13]
        minority_f1 = [metrics['per_class_f1'][i] for i in minority_classes]
        logger.info(f"  Minority Class F1: {[f'{i}:{f:.2f}' for i, f in zip(minority_classes, minority_f1)]}")
        
        # Save best model
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            patience_counter = 0
            
            model_name = f"{args.model}_multitask"
            if args.use_au_attention:
                model_name += "_attention"
            
            checkpoint_path = os.path.join(config['checkpoint_dir'], f"{model_name}_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'macro_f1': metrics['macro_f1'],
                'weighted_f1': metrics['weighted_f1'],
                'au_f1': au_metrics['au_mean_f1'],
                'per_class_metrics': metrics,
                'au_metrics': au_metrics,
                'config': config,
                'args': vars(args)
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path} ({metric_name}: {current_metric:.4f})")
        else:
            patience_counter += 1
            if patience_counter % 5 == 0:
                logger.info(f"Patience counter: {patience_counter}/{args.patience}")
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            logger.info(f"Best {metric_name}: {best_metric:.4f} at epoch {best_epoch}")
            break
        
        # Step scheduler
        scheduler.step()
    
    # Save final model
    model_name = f"{args.model}_multitask"
    if args.use_au_attention:
        model_name += "_attention"
    
    final_checkpoint_path = os.path.join(config['checkpoint_dir'], f"{model_name}_final.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'train_loss': train_loss,
        'macro_f1': metrics['macro_f1'],
        'weighted_f1': metrics['weighted_f1'],
        'au_f1': au_metrics['au_mean_f1'],
        'per_class_metrics': metrics,
        'au_metrics': au_metrics,
        'config': config,
        'args': vars(args)
    }, final_checkpoint_path)
    logger.info(f"Saved final model to {final_checkpoint_path}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'macro_f1s': macro_f1s,
        'au_f1s': au_f1s,
        'best_metric': best_metric,
        'best_epoch': best_epoch,
        'args': vars(args)
    }
    
    history_path = os.path.join(config['output_dir'], f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    logger.info(f"Saved training history to {history_path}")
    
    # Save per-class metrics report
    report = classification_report(val_labels, val_preds, output_dict=True, zero_division=0)
    report_path = os.path.join(config['output_dir'], f"{model_name}_classification_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    logger.info(f"Saved classification report to {report_path}")
    
    logger.info("=" * 60)
    logger.info("Training Completed")
    logger.info(f"Best {metric_name}: {best_metric:.4f} at epoch {best_epoch}")
    logger.info(f"Final Validation Accuracy: {val_acc:.2f}%")
    logger.info(f"Final Macro-F1: {metrics['macro_f1']:.4f}")
    logger.info(f"Final AU-F1: {au_metrics['au_mean_f1']:.4f}")
    logger.info(f"Train-Val Gap: {train_acc - val_acc:.2f}%")
    logger.info("=" * 60)
    
    return model, history

def main():
    """Main function for Kaggle multi-task training."""
    parser = argparse.ArgumentParser(description='Train emotion recognition model with AU supervision on Kaggle')
    parser.add_argument('--model', type=str, default='resnet50', 
                       choices=['resnet50', 'resnet101', 'vit'],
                       help='Model backbone')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--au_loss_weight', type=float, default=0.5,
                       help='Weight for AU loss (default: 0.5)')
    parser.add_argument('--augmentation', action='store_true', default=True,
                       help='Use data augmentation')
    parser.add_argument('--no_augmentation', action='store_false', dest='augmentation',
                       help='Disable data augmentation')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience (default: 15)')
    parser.add_argument('--monitor_metric', type=str, default='macro_f1',
                       choices=['accuracy', 'macro_f1', 'weighted_f1'],
                       help='Metric to monitor for early stopping (default: macro_f1)')
    parser.add_argument('--use_au_attention', action='store_true', default=False,
                       help='Use AU attention mechanism')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config()
    
    # Print configuration
    print("=" * 60)
    print("Kaggle Multi-Task Training Configuration")
    print("=" * 60)
    print(f"Running on Kaggle: {config['is_kaggle']}")
    print(f"Data Root (Aligned): {config['data_root_aligned']}")
    print(f"Output Directory: {config['output_dir']}")
    print(f"Checkpoint Directory: {config['checkpoint_dir']}")
    print(f"Log Directory: {config['log_dir']}")
    print(f"Model: {args.model}")
    print(f"Use AU Attention: {args.use_au_attention}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"AU Loss Weight: {args.au_loss_weight}")
    print(f"Augmentation: {args.augmentation}")
    print(f"Dropout Rate: {args.dropout_rate}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Patience: {args.patience}")
    print(f"Monitor Metric: {args.monitor_metric}")
    print("=" * 60)
    print()
    
    # Train model
    model, history = train_model(config, args)
    
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Best {args.monitor_metric}: {history['best_metric']:.4f}")
    print(f"Best Epoch: {history['best_epoch']}")
    print(f"Final Validation Accuracy: {history['val_accs'][-1]:.2f}%")
    print(f"Final Training Accuracy: {history['train_accs'][-1]:.2f}%")
    print(f"Final Macro-F1: {history['macro_f1s'][-1]:.4f}")
    print(f"Final AU-F1: {history['au_f1s'][-1]:.4f}")
    print(f"Train-Val Gap: {history['train_accs'][-1] - history['val_accs'][-1]:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
