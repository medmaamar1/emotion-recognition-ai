"""
Mixup and CutMix data augmentation techniques for regularization.
"""

import torch
import numpy as np
import torch.nn.functional as F


def mixup_data(x, y, alpha=0.2):
    """
    Apply Mixup augmentation to a batch of data.
    
    Mixup creates a linear interpolation between two samples and their labels.
    
    Args:
        x: Input batch of shape (N, C, H, W)
        y: Labels of shape (N,)
        alpha: Beta distribution parameter (default: 0.2)
    
    Returns:
        mixed_x: Mixed input batch
        y_a: Original labels
        y_b: Mixed labels
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute loss for Mixup-augmented data.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: Original labels
        y_b: Mixed labels
        lam: Mixing coefficient
    
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
    """
    Generate random bounding box for CutMix.
    
    Args:
        size: Input size (batch_size, C, H, W)
        lam: Lambda parameter
    
    Returns:
        Bounding box coordinates (bbx1, bby1, bbx2, bby2)
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    
    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    """
    Apply CutMix augmentation to a batch of data.
    
    CutMix cuts and pastes a patch from one image to another.
    
    Args:
        x: Input batch of shape (N, C, H, W)
        y: Labels of shape (N,)
        alpha: Beta distribution parameter (default: 1.0)
    
    Returns:
        mixed_x: Mixed input batch
        y_a: Original labels
        y_b: Mixed labels
        lam: Adjusted mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    # Generate random bounding box
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    
    # Apply cutmix
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda based on actual cut area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(2) * x.size(3)))
    
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute loss for CutMix-augmented data.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: Original labels
        y_b: Mixed labels
        lam: Mixing coefficient
    
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class MixupCutMixScheduler:
    """
    Scheduler for Mixup and CutMix augmentation.
    
    Gradually increases the probability of using Mixup/CutMix during training.
    
    Args:
        max_prob: Maximum probability of using Mixup/CutMix (default: 0.5)
        warmup_epochs: Number of epochs to reach max_prob (default: 5)
        total_epochs: Total training epochs (default: 100)
    """
    def __init__(self, max_prob=0.5, warmup_epochs=5, total_epochs=100):
        self.max_prob = max_prob
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
    def get_prob(self):
        """Get current probability of using Mixup/CutMix."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            return self.max_prob * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Constant after warmup
            return self.max_prob
    
    def step(self):
        """Advance to next epoch."""
        self.current_epoch += 1
    
    def should_apply(self):
        """Decide whether to apply Mixup/CutMix."""
        return np.random.random() < self.get_prob()
    
    def get_alpha(self):
        """Get current alpha value for beta distribution."""
        # Gradually increase alpha during training
        if self.current_epoch < self.warmup_epochs:
            return 0.1 + 0.1 * (self.current_epoch + 1) / self.warmup_epochs
        else:
            return 0.2


def apply_mixup_or_cutmix(x, y, scheduler, mode='mixup'):
    """
    Apply Mixup or CutMix based on scheduler decision.
    
    Args:
        x: Input batch of shape (N, C, H, W)
        y: Labels of shape (N,)
        scheduler: MixupCutMixScheduler instance
        mode: 'mixup' or 'cutmix' (default: 'mixup')
    
    Returns:
        Mixed data and labels, or original if not applied
    """
    if not scheduler.should_apply():
        return x, y, y, 1.0, False
    
    alpha = scheduler.get_alpha()
    
    if mode == 'mixup':
        return mixup_data(x, y, alpha=alpha) + (True,)
    elif mode == 'cutmix':
        return cutmix_data(x, y, alpha=alpha) + (True,)
    else:
        raise ValueError(f"Unknown mode: {mode}")


class MixupCutMixCollator:
    """
    Collator for Mixup/CutMix in DataLoader.
    
    Args:
        mode: 'mixup', 'cutmix', or 'auto' (default: 'auto')
        alpha: Beta distribution parameter (default: 0.2)
        prob: Probability of applying augmentation (default: 0.5)
    """
    def __init__(self, mode='auto', alpha=0.2, prob=0.5):
        self.mode = mode
        self.alpha = alpha
        self.prob = prob
        
    def __call__(self, batch):
        """
        Collate batch and apply Mixup/CutMix.
        
        Args:
            batch: List of samples from dataset
        
        Returns:
            Mixed batch
        """
        images = torch.stack([item['image'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        
        # Decide whether to apply augmentation
        if np.random.random() < self.prob:
            # Decide between mixup and cutmix
            if self.mode == 'auto':
                mode = np.random.choice(['mixup', 'cutmix'])
            else:
                mode = self.mode
            
            # Apply augmentation
            if mode == 'mixup':
                mixed_images, y_a, y_b, lam = mixup_data(images, labels, alpha=self.alpha)
            else:
                mixed_images, y_a, y_b, lam = cutmix_data(images, labels, alpha=self.alpha)
            
            return {
                'image': mixed_images,
                'label_a': y_a,
                'label_b': y_b,
                'lam': lam,
                'mixed': True,
                'mode': mode
            }
        else:
            return {
                'image': images,
                'label': labels,
                'mixed': False
            }


def mixup_criterion_multitask(criterion, pred, emotion_a, emotion_b, lam, au_a=None, au_b=None):
    """
    Compute loss for Mixup-augmented data with multi-task learning.
    
    Args:
        criterion: Loss function (should return total_loss, emotion_loss, au_loss)
        pred: Model predictions (emotion_pred, au_pred)
        emotion_a: Original emotion labels
        emotion_b: Mixed emotion labels
        lam: Mixing coefficient
        au_a: Original AU labels (optional)
        au_b: Mixed AU labels (optional)
    
    Returns:
        Mixed loss
    """
    emotion_pred, au_pred = pred
    
    # Compute loss for both label sets
    if au_a is not None and au_b is not None:
        total_loss_a, emo_loss_a, au_loss_a = criterion(emotion_pred, emotion_a, au_pred, au_a)
        total_loss_b, emo_loss_b, au_loss_b = criterion(emotion_pred, emotion_b, au_pred, au_b)
    else:
        # Single task
        total_loss_a = criterion(emotion_pred, emotion_a)
        total_loss_b = criterion(emotion_pred, emotion_b)
    
    # Mix losses
    mixed_loss = lam * total_loss_a + (1 - lam) * total_loss_b
    
    return mixed_loss


if __name__ == "__main__":
    # Test Mixup and CutMix
    print("Testing Mixup and CutMix...")
    
    # Create dummy data
    batch_size = 8
    num_classes = 14
    x = torch.randn(batch_size, 3, 224, 224)
    y = torch.randint(0, num_classes, (batch_size,))
    
    # Test Mixup
    print("\nTesting Mixup...")
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
    print(f"Mixup lambda: {lam:.4f}")
    print(f"Mixed shape: {mixed_x.shape}")
    print(f"Original labels: {y[:3]}")
    print(f"Mixed labels: {y_b[:3]}")
    
    # Test CutMix
    print("\nTesting CutMix...")
    mixed_x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)
    print(f"CutMix lambda: {lam:.4f}")
    print(f"Mixed shape: {mixed_x.shape}")
    
    # Test scheduler
    print("\nTesting MixupCutMixScheduler...")
    scheduler = MixupCutMixScheduler(max_prob=0.5, warmup_epochs=5, total_epochs=100)
    for epoch in [0, 2, 4, 5, 10, 50, 99]:
        scheduler.current_epoch = epoch
        prob = scheduler.get_prob()
        alpha = scheduler.get_alpha()
        print(f"Epoch {epoch}: prob={prob:.3f}, alpha={alpha:.3f}")
    
    # Test collator
    print("\nTesting MixupCutMixCollator...")
    collator = MixupCutMixCollator(mode='mixup', alpha=0.2, prob=1.0)
    batch = [{'image': torch.randn(3, 224, 224), 'label': torch.randint(0, 14, (1,)).item()} 
             for _ in range(8)]
    collated = collator(batch)
    print(f"Collated batch keys: {collated.keys()}")
    if collated['mixed']:
        print(f"Mixed: True, Mode: {collated['mode']}, Lambda: {collated['lam']:.4f}")
    
    print("\nAll Mixup and CutMix utilities tested successfully!")
