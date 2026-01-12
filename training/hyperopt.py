"""
Hyperparameter optimization framework using Optuna.
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Callable

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not available. Install with: pip install optuna")


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna.
    
    Args:
        objective_function: Function to optimize
        n_trials: Number of trials (default: 50)
        timeout: Timeout in seconds (default: None)
        direction: 'maximize' or 'minimize' (default: 'maximize')
        study_name: Name of the study (default: None)
        storage: Database storage URL (default: None)
    """
    def __init__(self, objective_function: Callable, n_trials: int = 50,
                 timeout: float = None, direction: str = 'maximize',
                 study_name: str = None, storage: str = None):
        self.objective_function = objective_function
        self.n_trials = n_trials
        self.timeout = timeout
        self.direction = direction
        self.study_name = study_name
        self.storage = storage
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna is required for hyperparameter optimization. "
                          "Install with: pip install optuna")
        
        self.study = None
        self.best_params = None
        self.best_value = None
    
    def create_study(self):
        """Create Optuna study."""
        self.study = optuna.create_study(
            direction=self.direction,
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True
        )
    
    def optimize(self):
        """Run hyperparameter optimization."""
        if self.study is None:
            self.create_study()
        
        print(f"Starting hyperparameter optimization with {self.n_trials} trials...")
        
        self.study.optimize(
            self.objective_function,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        print(f"\nBest value: {self.best_value:.4f}")
        print(f"Best parameters: {self.best_params}")
        
        return self.best_params, self.best_value
    
    def get_best_params(self):
        """Get best hyperparameters."""
        return self.best_params
    
    def get_best_value(self):
        """Get best objective value."""
        return self.best_value
    
    def get_trials_dataframe(self):
        """Get all trials as a DataFrame."""
        if self.study is None:
            return None
        return self.study.trials_dataframe()
    
    def save_results(self, filepath):
        """Save optimization results to file."""
        if self.study is None:
            print("No study to save.")
            return
        
        # Save best parameters
        results = {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': len(self.study.trials),
            'direction': self.direction
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {filepath}")


def suggest_hyperparameters(trial, search_space: Dict[str, Any]):
    """
    Suggest hyperparameters from search space.
    
    Args:
        trial: Optuna trial object
        search_space: Dictionary defining the search space
    
    Returns:
        Dictionary of suggested hyperparameters
    """
    params = {}
    
    for param_name, param_config in search_space.items():
        param_type = param_config['type']
        
        if param_type == 'float':
            if 'log' in param_config and param_config['log']:
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=True
                )
            else:
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high']
                )
        
        elif param_type == 'int':
            if 'log' in param_config and param_config['log']:
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=True
                )
            else:
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high']
                )
        
        elif param_type == 'categorical':
            params[param_name] = trial.suggest_categorical(
                param_name,
                param_config['choices']
            )
        
        elif param_type == 'discrete_uniform':
            params[param_name] = trial.suggest_discrete_uniform(
                param_name,
                param_config['low'],
                param_config['high'],
                param_config['q']
            )
    
    return params


def get_default_search_space():
    """
    Get default search space for emotion recognition.
    
    Returns:
        Dictionary defining the search space
    """
    return {
        'learning_rate': {
            'type': 'float',
            'low': 1e-5,
            'high': 1e-2,
            'log': True
        },
        'batch_size': {
            'type': 'categorical',
            'choices': [16, 32, 64]
        },
        'weight_decay': {
            'type': 'float',
            'low': 1e-6,
            'high': 1e-3,
            'log': True
        },
        'dropout': {
            'type': 'float',
            'low': 0.1,
            'high': 0.5
        },
        'mixup_alpha': {
            'type': 'float',
            'low': 0.1,
            'high': 0.5
        },
        'label_smoothing': {
            'type': 'float',
            'low': 0.0,
            'high': 0.2
        },
        'focal_gamma': {
            'type': 'float',
            'low': 1.0,
            'high': 3.0
        },
        'warmup_epochs': {
            'type': 'int',
            'low': 3,
            'high': 10
        }
    }


def create_objective_function(model_class, train_loader, val_loader, device,
                         num_classes=14, num_epochs=20):
    """
    Create objective function for hyperparameter optimization.
    
    Args:
        model_class: Model class to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        num_classes: Number of classes (default: 14)
        num_epochs: Number of training epochs (default: 20)
    
    Returns:
        Objective function for Optuna
    """
    def objective(trial):
        # Suggest hyperparameters
        search_space = get_default_search_space()
        params = suggest_hyperparameters(trial, search_space)
        
        # Create model
        model = model_class(num_classes=num_classes, dropout=params['dropout']).to(device)
        
        # Loss function
        from utils.losses import FocalLoss, LabelSmoothingCrossEntropy
        if params['label_smoothing'] > 0:
            criterion = LabelSmoothingCrossEntropy(smoothing=params['label_smoothing'])
        else:
            criterion = FocalLoss(gamma=params['focal_gamma'])
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # Scheduler
        from training.schedulers import CosineAnnealingWarmupScheduler
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer,
            warmup_epochs=params['warmup_epochs'],
            max_epochs=num_epochs
        )
        
        # Training loop
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
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
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
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
            
            val_acc = 100 * val_correct / val_total
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            scheduler.step()
            
            # Report intermediate value
            trial.report(val_acc, epoch)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_val_acc
    
    return objective


def optimize_hyperparameters(model_class, train_loader, val_loader, device,
                          n_trials=50, num_epochs=20):
    """
    Run hyperparameter optimization.
    
    Args:
        model_class: Model class to optimize
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        n_trials: Number of trials (default: 50)
        num_epochs: Number of training epochs (default: 20)
    
    Returns:
        Best hyperparameters and best value
    """
    # Create objective function
    objective = create_objective_function(
        model_class, train_loader, val_loader, device,
        num_classes=14, num_epochs=num_epochs
    )
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        objective_function=objective,
        n_trials=n_trials,
        direction='maximize',
        study_name='emotion_recognition_optimization'
    )
    
    # Run optimization
    best_params, best_value = optimizer.optimize()
    
    return best_params, best_value


if __name__ == "__main__":
    # Test hyperparameter optimization
    print("Testing hyperparameter optimization framework...")
    
    if not OPTUNA_AVAILABLE:
        print("optuna not available. Install with: pip install optuna")
    else:
        # Test suggest_hyperparameters
        print("\nTesting suggest_hyperparameters...")
        study = optuna.create_study(direction='maximize')
        trial = study.ask()
        
        search_space = get_default_search_space()
        params = suggest_hyperparameters(trial, search_space)
        print(f"Suggested parameters: {params}")
        
        # Test HyperparameterOptimizer
        print("\nTesting HyperparameterOptimizer...")
        
        def dummy_objective(trial):
            # Suggest parameters
            search_space = get_default_search_space()
            params = suggest_hyperparameters(trial, search_space)
            
            # Simulate training
            value = np.random.normal(0.7, 0.1)
            
            return value
        
        optimizer = HyperparameterOptimizer(
            objective_function=dummy_objective,
            n_trials=10,
            direction='maximize',
            study_name='test_study'
        )
        
        best_params, best_value = optimizer.optimize()
        print(f"Best value: {best_value:.4f}")
        print(f"Best parameters: {best_params}")
        
        # Test save results
        print("\nTesting save_results...")
        optimizer.save_results('test_hyperopt_results.json')
        
        print("\nAll hyperparameter optimization utilities tested successfully!")
