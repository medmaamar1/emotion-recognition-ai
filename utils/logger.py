"""
Comprehensive logging and experiment tracking utilities.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


def setup_logger(log_dir, name, level=logging.INFO):
    """
    Setup a logger with file and console handlers.
    
    Args:
        log_dir: Directory to save log files
        name: Name of the logger
        level: Logging level (default: logging.INFO)
    
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = os.path.join(log_dir, f'{name}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


class ExperimentTracker:
    """
    Comprehensive experiment tracking with TensorBoard and optional Weights & Biases.
    
    Args:
        project_name: Name of the project
        experiment_name: Name of the experiment
        config: Configuration dictionary
        use_wandb: Whether to use Weights & Biases (default: False)
        log_dir: Directory for TensorBoard logs (default: './runs')
    """
    def __init__(self, project_name, experiment_name, config, use_wandb=False, log_dir='./runs'):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config
        self.use_wandb = use_wandb
        self.log_dir = log_dir
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize TensorBoard
        self.writer = SummaryWriter(self.experiment_dir)
        
        # Initialize Weights & Biases
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.init(project=project_name, name=experiment_name, config=config)
        
        # Save configuration
        self.save_config()
        
        # Metrics history
        self.metrics_history = {}
        self.start_time = time.time()
    
    def save_config(self):
        """Save experiment configuration to JSON file."""
        config_path = os.path.join(self.experiment_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def log_metrics(self, metrics, step=None, prefix=''):
        """
        Log metrics to TensorBoard and optionally Weights & Biases.
        
        Args:
            metrics: Dictionary of metrics
            step: Step number (default: None)
            prefix: Prefix for metric names (default: '')
        """
        for key, value in metrics.items():
            metric_name = f"{prefix}/{key}" if prefix else key
            
            # Log to TensorBoard
            if step is not None:
                self.writer.add_scalar(metric_name, value, step)
            else:
                self.writer.add_scalar(metric_name, value)
            
            # Log to Weights & Biases
            if self.use_wandb and WANDB_AVAILABLE:
                wandb.log({metric_name: value}, step=step)
            
            # Store in history
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            self.metrics_history[metric_name].append({
                'value': value,
                'step': step,
                'time': time.time() - self.start_time
            })
    
    def log_metrics_dict(self, metrics_dict, step=None):
        """
        Log nested metrics dictionary.
        
        Args:
            metrics_dict: Nested dictionary of metrics
            step: Step number (default: None)
        """
        for prefix, metrics in metrics_dict.items():
            self.log_metrics(metrics, step=step, prefix=prefix)
    
    def log_image(self, tag, image, step=None):
        """
        Log image to TensorBoard.
        
        Args:
            tag: Image tag
            image: Image tensor or numpy array
            step: Step number (default: None)
        """
        if step is not None:
            self.writer.add_image(tag, image, step)
        else:
            self.writer.add_image(tag, image)
    
    def log_images(self, tag, images, step=None):
        """
        Log multiple images to TensorBoard.
        
        Args:
            tag: Image tag
            images: List of image tensors or numpy arrays
            step: Step number (default: None)
        """
        if step is not None:
            self.writer.add_images(tag, images, step)
        else:
            self.writer.add_images(tag, images)
    
    def log_histogram(self, tag, values, step=None):
        """
        Log histogram to TensorBoard.
        
        Args:
            tag: Histogram tag
            values: Values to histogram
            step: Step number (default: None)
        """
        if step is not None:
            self.writer.add_histogram(tag, values, step)
        else:
            self.writer.add_histogram(tag, values)
    
    def log_model(self, model, name):
        """
        Log model architecture and parameters.
        
        Args:
            model: PyTorch model
            name: Model name
        """
        # Log model architecture
        model_path = os.path.join(self.experiment_dir, f'{name}_architecture.txt')
        with open(model_path, 'w') as f:
            f.write(str(model))
        
        # Log parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        params_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }
        
        self.log_metrics(params_info, prefix=f'model/{name}')
        
        # Log to Weights & Biases
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.watch(model, log_freq=100)
    
    def log_confusion_matrix(self, y_true, y_pred, class_names, step=None):
        """
        Log confusion matrix to TensorBoard.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            step: Step number (default: None)
        """
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, 
                   yticklabels=class_names, cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        # Log to TensorBoard
        self.writer.add_figure('confusion_matrix', fig, step)
        plt.close(fig)
        
        # Log to Weights & Biases
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.log({'confusion_matrix': wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names
            )}, step=step)
    
    def log_learning_rate(self, optimizer, step=None):
        """
        Log learning rate.
        
        Args:
            optimizer: PyTorch optimizer
            step: Step number (default: None)
        """
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            self.log_metrics({'learning_rate': lr}, step=step, prefix=f'optimizer/group_{i}')
    
    def log_gradient_norm(self, model, step=None):
        """
        Log gradient norms.
        
        Args:
            model: PyTorch model
            step: Step number (default: None)
        """
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        self.log_metrics({'gradient_norm': total_norm}, step=step, prefix='training')
    
    def log_checkpoint(self, model, optimizer, epoch, metrics, filename='checkpoint.pth'):
        """
        Save model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch
            metrics: Dictionary of metrics
            filename: Checkpoint filename (default: 'checkpoint.pth')
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(self.experiment_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        
        # Log to Weights & Biases
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.save(checkpoint_path)
    
    def log_best_model(self, model, metrics, filename='best_model.pth'):
        """
        Save best model checkpoint.
        
        Args:
            model: PyTorch model
            metrics: Dictionary of metrics
            filename: Model filename (default: 'best_model.pth')
        """
        model_path = os.path.join(self.experiment_dir, filename)
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'config': self.config
        }, model_path)
        
        # Log to Weights & Biases
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.save(model_path)
    
    def get_best_metric(self, metric_name, mode='max'):
        """
        Get best value for a metric from history.
        
        Args:
            metric_name: Name of the metric
            mode: 'max' or 'min' (default: 'max')
        
        Returns:
            Best value and step
        """
        if metric_name not in self.metrics_history:
            return None, None
        
        values = [(m['value'], m['step']) for m in self.metrics_history[metric_name]]
        
        if mode == 'max':
            best_value, best_step = max(values, key=lambda x: x[0])
        else:
            best_value, best_step = min(values, key=lambda x: x[0])
        
        return best_value, best_step
    
    def save_metrics_history(self):
        """Save metrics history to JSON file."""
        history_path = os.path.join(self.experiment_dir, 'metrics_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
    
    def finish(self):
        """Finish experiment and close writers."""
        self.writer.close()
        
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        self.save_metrics_history()
        
        elapsed_time = time.time() - self.start_time
        print(f"Experiment finished in {elapsed_time:.2f} seconds")
        print(f"Results saved to: {self.experiment_dir}")


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Args:
        patience: Number of epochs to wait before stopping (default: 10)
        min_delta: Minimum change to qualify as improvement (default: 0.0)
        mode: 'min' or 'max' (default: 'min')
        verbose: Whether to print messages (default: True)
    """
    def __init__(self, patience=10, min_delta=0.0, mode='min', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = lambda x, y: x < y - min_delta
        else:
            self.monitor_op = lambda x, y: x > y + min_delta
    
    def __call__(self, metric):
        """
        Check if should stop training.
        
        Args:
            metric: Current metric value
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = metric
            return False
        
        if self.monitor_op(metric, self.best_score):
            self.best_score = metric
            self.counter = 0
            if self.verbose:
                print(f'Validation improved: {self.best_score:.4f}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'Early stopping triggered!')
        
        return self.early_stop


class ModelCheckpoint:
    """
    Model checkpointing with best model saving.
    
    Args:
        save_dir: Directory to save checkpoints
        monitor: Metric to monitor (default: 'val_loss')
        mode: 'min' or 'max' (default: 'min')
        save_best_only: Whether to save only the best model (default: True)
        verbose: Whether to print messages (default: True)
    """
    def __init__(self, save_dir, monitor='val_loss', mode='min', 
                 save_best_only=True, verbose=True):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.best_score = None
        self.epoch = 0
        
        if mode == 'min':
            self.monitor_op = lambda x, y: x < y
        else:
            self.monitor_op = lambda x, y: x > y
    
    def __call__(self, model, optimizer, epoch, metrics):
        """
        Save checkpoint if metric improved.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch
            metrics: Dictionary of metrics
        """
        self.epoch = epoch
        current_score = metrics.get(self.monitor)
        
        if current_score is None:
            if self.verbose:
                print(f'Metric {self.monitor} not found in metrics')
            return
        
        if self.best_score is None or self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self._save_checkpoint(model, optimizer, epoch, metrics, 'best_model.pth')
            if self.verbose:
                print(f'New best {self.monitor}: {current_score:.4f}')
        
        if not self.save_best_only:
            self._save_checkpoint(model, optimizer, epoch, metrics, f'checkpoint_epoch_{epoch}.pth')
    
    def _save_checkpoint(self, model, optimizer, epoch, metrics, filename):
        """Save checkpoint to file."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_score': self.best_score
        }
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        
        if self.verbose:
            print(f'Checkpoint saved: {filepath}')


if __name__ == "__main__":
    # Test experiment tracking
    print("Testing experiment tracking utilities...")
    
    # Test ExperimentTracker
    print("\nTesting ExperimentTracker...")
    config = {
        'model': 'resnet50',
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 100
    }
    
    tracker = ExperimentTracker(
        project_name='emotion_recognition',
        experiment_name='test_experiment',
        config=config,
        use_wandb=False
    )
    
    # Log some metrics
    for epoch in range(5):
        metrics = {
            'train_loss': 1.0 - epoch * 0.1,
            'val_loss': 1.2 - epoch * 0.08,
            'train_acc': 0.5 + epoch * 0.08,
            'val_acc': 0.45 + epoch * 0.07
        }
        tracker.log_metrics(metrics, step=epoch)
        tracker.log_metrics_dict({
            'training': {'loss': metrics['train_loss'], 'accuracy': metrics['train_acc']},
            'validation': {'loss': metrics['val_loss'], 'accuracy': metrics['val_acc']}
        }, step=epoch)
    
    # Test EarlyStopping
    print("\nTesting EarlyStopping...")
    early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode='min')
    metrics = [1.0, 0.9, 0.95, 0.85, 0.9, 0.88, 0.87, 0.89]
    for metric in metrics:
        should_stop = early_stopping(metric)
        print(f"Metric: {metric:.2f}, Stop: {should_stop}")
        if should_stop:
            break
    
    # Test ModelCheckpoint
    print("\nTesting ModelCheckpoint...")
    import tempfile
    import torch.nn as nn
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = ModelCheckpoint(tmpdir, monitor='val_loss', mode='min', verbose=True)
        
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)
        
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        for epoch in range(5):
            metrics = {'val_loss': 1.0 - epoch * 0.1, 'val_acc': 0.5 + epoch * 0.05}
            checkpoint(model, optimizer, epoch, metrics)
    
    tracker.finish()
    print("\nAll experiment tracking utilities tested successfully!")
