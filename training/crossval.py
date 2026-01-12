"""
Cross-validation for robust model evaluation.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm


class CrossValidator:
    """
    Cross-validation for model evaluation.
    
    Args:
        dataset: PyTorch dataset
        k: Number of folds (default: 5)
        shuffle: Whether to shuffle data (default: True)
        stratified: Whether to use stratified sampling (default: True)
        random_state: Random seed (default: 42)
    """
    def __init__(self, dataset, k=5, shuffle=True, stratified=True, random_state=42):
        self.dataset = dataset
        self.k = k
        self.shuffle = shuffle
        self.stratified = stratified
        self.random_state = random_state
        
        # Get labels for stratified sampling
        if stratified and hasattr(dataset, 'emotions'):
            labels = [dataset.emotions[img_id] for img_id in dataset.image_ids]
            self.labels = np.array(labels)
        else:
            self.labels = None
        
        # Create KFold splitter
        if stratified and self.labels is not None:
            self.kfold = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)
        else:
            self.kfold = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
        
        # Generate fold indices
        self.fold_indices = list(self.kfold.split(range(len(dataset)), self.labels))
    
    def get_fold(self, fold_idx):
        """
        Get train and validation indices for a specific fold.
        
        Args:
            fold_idx: Fold index (0 to k-1)
        
        Returns:
            Tuple of (train_indices, val_indices)
        """
        if fold_idx >= self.k:
            raise ValueError(f"fold_idx must be less than {self.k}")
        
        train_idx, val_idx = self.fold_indices[fold_idx]
        return train_idx, val_idx
    
    def get_fold_datasets(self, fold_idx):
        """
        Get train and validation datasets for a specific fold.
        
        Args:
            fold_idx: Fold index (0 to k-1)
        
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        train_idx, val_idx = self.get_fold(fold_idx)
        train_dataset = Subset(self.dataset, train_idx)
        val_dataset = Subset(self.dataset, val_idx)
        return train_dataset, val_dataset
    
    def get_fold_loaders(self, fold_idx, batch_size=32, num_workers=0, 
                        train_transform=None, val_transform=None):
        """
        Get train and validation data loaders for a specific fold.
        
        Args:
            fold_idx: Fold index (0 to k-1)
            batch_size: Batch size (default: 32)
            num_workers: Number of workers (default: 0)
            train_transform: Transform for training data (default: None)
            val_transform: Transform for validation data (default: None)
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        train_dataset, val_dataset = self.get_fold_datasets(fold_idx)
        
        # Apply transforms if provided
        if train_transform is not None:
            train_dataset = TransformSubset(train_dataset, train_transform)
        if val_transform is not None:
            val_dataset = TransformSubset(val_dataset, val_transform)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return train_loader, val_loader


class TransformSubset(Subset):
    """
    Subset with transform support.
    
    Args:
        dataset: Original dataset
        indices: Indices to include
        transform: Transform to apply (default: None)
    """
    def __init__(self, dataset, indices, transform=None):
        super(TransformSubset, self).__init__(dataset, indices)
        self.transform = transform
    
    def __getitem__(self, idx):
        item = self.dataset[self.indices[idx]]
        
        if self.transform is not None:
            if isinstance(item, dict):
                item['image'] = self.transform(item['image'])
            else:
                item = self.transform(item)
        
        return item


def cross_validate(model_class, dataset, device, k=5, epochs=20, 
                  batch_size=32, learning_rate=0.001, **model_kwargs):
    """
    Perform k-fold cross-validation.
    
    Args:
        model_class: Model class to train
        dataset: PyTorch dataset
        device: Device to train on
        k: Number of folds (default: 5)
        epochs: Number of training epochs (default: 20)
        batch_size: Batch size (default: 32)
        learning_rate: Learning rate (default: 0.001)
        **model_kwargs: Additional arguments for model initialization
    
    Returns:
        Dictionary with cross-validation results
    """
    from utils.losses import FocalLoss
    from training.schedulers import CosineAnnealingWarmupScheduler
    from evaluation.metrics import FEREvaluator
    
    # Create cross-validator
    cv = CrossValidator(dataset, k=k, stratified=True)
    
    # Store results
    fold_results = []
    all_predictions = []
    all_labels = []
    
    # Perform cross-validation
    for fold_idx in range(k):
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx + 1}/{k}")
        print(f"{'='*50}")
        
        # Get fold data loaders
        train_loader, val_loader = cv.get_fold_loaders(
            fold_idx,
            batch_size=batch_size,
            train_transform=None,  # Use dataset's transform
            val_transform=None
        )
        
        # Create model
        model = model_class(**model_kwargs).to(device)
        
        # Loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer,
            warmup_epochs=5,
            max_epochs=epochs
        )
        
        # Training loop
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            fold_predictions = []
            fold_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    fold_predictions.extend(predicted.cpu().numpy())
                    fold_labels.extend(labels.cpu().numpy())
            
            val_acc = 100 * val_correct / val_total
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            scheduler.step()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        print(f"Fold {fold_idx + 1} Best Val Acc: {best_val_acc:.2f}%")
        
        # Store fold results
        fold_results.append(best_val_acc)
        all_predictions.extend(fold_predictions)
        all_labels.extend(fold_labels)
    
    # Compute overall results
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    
    # Compute overall metrics
    evaluator = FEREvaluator()
    metrics = evaluator.calculate_metrics(all_labels, all_predictions)
    
    results = {
        'fold_accuracies': fold_results,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'f1_macro': metrics['f1_macro'],
        'f1_weighted': metrics['f1_weighted'],
        'full_report': metrics['full_report'],
        'all_predictions': all_predictions,
        'all_labels': all_labels
    }
    
    return results


def nested_cross_validate(model_class, dataset, device, outer_k=5, inner_k=3,
                       epochs=20, batch_size=32, learning_rate=0.001,
                       **model_kwargs):
    """
    Perform nested cross-validation for hyperparameter tuning.
    
    Args:
        model_class: Model class to train
        dataset: PyTorch dataset
        device: Device to train on
        outer_k: Number of outer folds (default: 5)
        inner_k: Number of inner folds (default: 3)
        epochs: Number of training epochs (default: 20)
        batch_size: Batch size (default: 32)
        learning_rate: Learning rate (default: 0.001)
        **model_kwargs: Additional arguments for model initialization
    
    Returns:
        Dictionary with nested cross-validation results
    """
    # Create outer cross-validator
    outer_cv = CrossValidator(dataset, k=outer_k, stratified=True)
    
    # Store results
    outer_results = []
    
    # Perform nested cross-validation
    for outer_fold in range(outer_k):
        print(f"\n{'='*50}")
        print(f"Outer Fold {outer_fold + 1}/{outer_k}")
        print(f"{'='*50}")
        
        # Get outer fold indices
        outer_train_idx, outer_test_idx = outer_cv.get_fold(outer_fold)
        
        # Create inner dataset for hyperparameter tuning
        inner_dataset = Subset(dataset, outer_train_idx)
        
        # Perform inner cross-validation
        inner_results = cross_validate(
            model_class,
            inner_dataset,
            device,
            k=inner_k,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            **model_kwargs
        )
        
        outer_results.append(inner_results)
    
    # Compute overall results
    mean_f1_macro = np.mean([r['f1_macro'] for r in outer_results])
    std_f1_macro = np.std([r['f1_macro'] for r in outer_results])
    
    results = {
        'outer_results': outer_results,
        'mean_f1_macro': mean_f1_macro,
        'std_f1_macro': std_f1_macro
    }
    
    return results


if __name__ == "__main__":
    # Test cross-validation
    print("Testing cross-validation utilities...")
    
    # Create dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.size = size
            self.emotions = {i: i % 14 for i in range(size)}
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                'image': torch.randn(3, 224, 224),
                'label': idx % 14,
                'image_id': str(idx)
            }
    
    dataset = DummyDataset(size=100)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test CrossValidator
    print("\nTesting CrossValidator...")
    cv = CrossValidator(dataset, k=5, stratified=True)
    print(f"Number of folds: {cv.k}")
    
    train_idx, val_idx = cv.get_fold(0)
    print(f"Fold 0 - Train size: {len(train_idx)}, Val size: {len(val_idx)}")
    
    train_dataset, val_dataset = cv.get_fold_datasets(0)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Test TransformSubset
    print("\nTesting TransformSubset...")
    from torchvision import transforms
    
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_subset = TransformSubset(train_dataset, train_idx, transform=transform)
    sample = transform_subset[0]
    print(f"Transformed sample keys: {sample.keys()}")
    
    # Test cross_validate (with dummy model)
    print("\nTesting cross_validate...")
    
    class DummyModel(torch.nn.Module):
        def __init__(self, num_classes=14):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(16, num_classes)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    results = cross_validate(
        DummyModel,
        dataset,
        device,
        k=3,
        epochs=5,
        batch_size=16,
        learning_rate=0.001,
        num_classes=14
    )
    
    print(f"Mean accuracy: {results['mean_accuracy']:.2f}%")
    print(f"Std accuracy: {results['std_accuracy']:.2f}%")
    print(f"F1 Macro: {results['f1_macro']:.4f}")
    print(f"F1 Weighted: {results['f1_weighted']:.4f}")
    
    print("\nAll cross-validation utilities tested successfully!")
