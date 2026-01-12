"""
Advanced learning rate schedulers for improved training.
"""

import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupScheduler(_LRScheduler):
    """
    Cosine annealing learning rate scheduler with warmup.
    
    Combines linear warmup with cosine annealing for smooth LR transitions.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs (default: 5)
        max_epochs: Total number of training epochs (default: 100)
        min_lr: Minimum learning rate (default: 0)
        warmup_start_lr: Starting learning rate for warmup (default: 0)
        last_epoch: Last epoch index (default: -1)
    """
    def __init__(self, optimizer, warmup_epochs=5, max_epochs=100, 
                 min_lr=0, warmup_start_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        super(CosineAnnealingWarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * alpha 
                    for _ in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.min_lr + (self.base_lr - self.min_lr) * 
                    0.5 * (1 + np.cos(np.pi * progress)) for _ in self.base_lrs]


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine annealing with warmup and restarts.
    
    Args:
        optimizer: PyTorch optimizer
        first_cycle_steps: Number of steps in first cycle
        cycle_mult: Cycle multiplier (default: 1.0)
        max_lr: Maximum learning rate (default: 0.1)
        min_lr: Minimum learning rate (default: 0.001)
        warmup_steps: Number of warmup steps (default: 0)
        gamma: Decay factor for max_lr after each cycle (default: 1.0)
        last_epoch: Last epoch index (default: -1)
    """
    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1.0, 
                 max_lr=0.1, min_lr=0.001, warmup_steps=0, gamma=1.0, last_epoch=-1):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.step_in_cycle < self.warmup_steps:
            # Warmup phase
            alpha = self.step_in_cycle / self.warmup_steps
            return [self.min_lr + (self.max_lr - self.min_lr) * alpha 
                    for _ in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps)
            return [self.min_lr + (self.max_lr - self.min_lr) * 
                    0.5 * (1 + np.cos(np.pi * progress)) for _ in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(np.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** n
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
        
        self.last_epoch = epoch
        self._step()
        
        if self.step_in_cycle >= self.cur_cycle_steps:
            self.cycle += 1
            self.step_in_cycle = 0
            self.cur_cycle_steps = int(self.cur_cycle_steps * self.cycle_mult)
            self.max_lr *= self.gamma


class LinearWarmupScheduler(_LRScheduler):
    """
    Simple linear warmup scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs (default: 5)
        target_lr: Target learning rate after warmup (default: None)
        start_lr: Starting learning rate (default: 0)
        last_epoch: Last epoch index (default: -1)
    """
    def __init__(self, optimizer, warmup_epochs=5, target_lr=None, start_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr
        self.target_lr = target_lr if target_lr else optimizer.param_groups[0]['lr']
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [self.start_lr + (self.target_lr - self.start_lr) * alpha 
                    for _ in self.base_lrs]
        else:
            return [self.target_lr for _ in self.base_lrs]


class PolynomialDecayScheduler(_LRScheduler):
    """
    Polynomial decay learning rate scheduler with warmup.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps (default: 1000)
        total_steps: Total training steps (default: 100000)
        end_lr: Final learning rate (default: 0)
        power: Polynomial power (default: 1.0)
        last_epoch: Last epoch index (default: -1)
    """
    def __init__(self, optimizer, warmup_steps=1000, total_steps=100000, 
                 end_lr=0, power=1.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.end_lr = end_lr
        self.power = power
        self.base_lr = optimizer.param_groups[0]['lr']
        super(PolynomialDecayScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            return [self.base_lr * (self.last_epoch + 1) / self.warmup_steps 
                    for _ in self.base_lrs]
        else:
            # Polynomial decay phase
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [self.end_lr + (self.base_lr - self.end_lr) * (1 - progress) ** self.power 
                    for _ in self.base_lrs]


class OneCycleLR(_LRScheduler):
    """
    One Cycle learning rate policy.
    
    Args:
        optimizer: PyTorch optimizer
        max_lr: Maximum learning rate
        total_steps: Total number of training steps
        pct_start: Percentage of total steps for increasing phase (default: 0.3)
        anneal_strategy: 'cos' or 'linear' (default: 'cos')
        div_factor: Initial lr division factor (default: 25.0)
        final_div_factor: Final lr division factor (default: 1e4)
        last_epoch: Last epoch index (default: -1)
    """
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3, 
                 anneal_strategy='cos', div_factor=25.0, final_div_factor=1e4, last_epoch=-1):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        
        self.step_size_up = int(total_steps * pct_start)
        self.step_size_down = total_steps - self.step_size_up
        
        super(OneCycleLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.step_size_up:
            # Increasing phase
            progress = self.last_epoch / self.step_size_up
            if self.anneal_strategy == 'cos':
                return [self.initial_lr + (self.max_lr - self.initial_lr) * 
                        (1 - np.cos(np.pi * progress)) / 2 for _ in self.base_lrs]
            else:
                return [self.initial_lr + (self.max_lr - self.initial_lr) * progress 
                        for _ in self.base_lrs]
        elif self.last_epoch < self.total_steps:
            # Decreasing phase
            progress = (self.last_epoch - self.step_size_up) / self.step_size_down
            if self.anneal_strategy == 'cos':
                return [self.max_lr + (self.final_lr - self.max_lr) * 
                        (1 + np.cos(np.pi * progress)) / 2 for _ in self.base_lrs]
            else:
                return [self.max_lr + (self.final_lr - self.max_lr) * progress 
                        for _ in self.base_lrs]
        else:
            return [self.final_lr for _ in self.base_lrs]


class ReduceLROnPlateauWithWarmup:
    """
    Reduce LR on plateau with initial warmup.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs (default: 5)
        mode: 'min' or 'max' (default: 'min')
        factor: Reduction factor (default: 0.1)
        patience: Number of epochs with no improvement (default: 10)
        threshold: Threshold for measuring improvement (default: 1e-4)
        min_lr: Minimum learning rate (default: 0)
    """
    def __init__(self, optimizer, warmup_epochs=5, mode='min', factor=0.1, 
                 patience=10, threshold=1e-4, min_lr=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
        self.best = None
        self.num_bad_epochs = 0
        self.warmup_start_lr = self.base_lr / 10
        
    def step(self, metric):
        """Update learning rate based on metric."""
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            alpha = (self.current_epoch + 1) / self.warmup_epochs
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * alpha
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Plateau phase
            if self.best is None:
                self.best = metric
            elif (self.mode == 'min' and metric < self.best - self.threshold) or \
                 (self.mode == 'max' and metric > self.best + self.threshold):
                self.best = metric
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
            
            if self.num_bad_epochs >= self.patience:
                for param_group in self.optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    param_group['lr'] = new_lr
                self.num_bad_epochs = 0
        
        self.current_epoch += 1
    
    def get_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


def get_scheduler(optimizer, scheduler_type='cosine_warmup', **kwargs):
    """
    Factory function to get learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler (default: 'cosine_warmup')
        **kwargs: Additional arguments for scheduler
    
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == 'cosine_warmup':
        return CosineAnnealingWarmupScheduler(optimizer, **kwargs)
    elif scheduler_type == 'cosine_warmup_restarts':
        return CosineAnnealingWarmupRestarts(optimizer, **kwargs)
    elif scheduler_type == 'linear_warmup':
        return LinearWarmupScheduler(optimizer, **kwargs)
    elif scheduler_type == 'polynomial_decay':
        return PolynomialDecayScheduler(optimizer, **kwargs)
    elif scheduler_type == 'onecycle':
        return OneCycleLR(optimizer, **kwargs)
    elif scheduler_type == 'reduce_on_plateau':
        return ReduceLROnPlateauWithWarmup(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


if __name__ == "__main__":
    # Test schedulers
    import matplotlib.pyplot as plt
    
    print("Testing learning rate schedulers...")
    
    # Create a simple optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Test Cosine Annealing with Warmup
    print("\nTesting CosineAnnealingWarmupScheduler...")
    scheduler = CosineAnnealingWarmupScheduler(optimizer, warmup_epochs=5, max_epochs=50)
    lrs = []
    for epoch in range(50):
        lr = scheduler.get_lr()[0]
        lrs.append(lr)
        scheduler.step()
    print(f"LR range: {min(lrs):.6f} to {max(lrs):.6f}")
    
    # Test One Cycle
    print("\nTesting OneCycleLR...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=100, pct_start=0.3)
    lrs = []
    for step in range(100):
        lr = scheduler.get_lr()[0]
        lrs.append(lr)
        scheduler.step()
    print(f"LR range: {min(lrs):.6f} to {max(lrs):.6f}")
    
    # Test Reduce on Plateau
    print("\nTesting ReduceLROnPlateauWithWarmup...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateauWithWarmup(optimizer, warmup_epochs=5, mode='min', patience=3)
    lrs = []
    metrics = [0.9, 0.8, 0.7, 0.75, 0.65, 0.6, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7]
    for metric in metrics:
        scheduler.step(metric)
        lrs.append(scheduler.get_lr())
    print(f"LR changes: {lrs[:5]} ... {lrs[-5:]}")
    
    print("\nAll schedulers tested successfully!")
