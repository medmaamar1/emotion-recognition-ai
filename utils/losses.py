"""
Advanced loss functions for emotion recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss = -alpha * (1 - pt)^gamma * log(pt)
    where pt is the model's estimated probability for the correct class.
    
    Args:
        alpha: Weighting factor for rare classes (default: None)
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method (default: 'mean')
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits) of shape (N, C)
            targets: Ground truth labels of shape (N,)
        
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha = torch.tensor(self.alpha, dtype=inputs.dtype, device=inputs.device)
            else:
                alpha = self.alpha
            focal_loss = alpha[targets] * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss.
    
    Smooths the target distribution to prevent overconfidence and improve generalization.
    
    Args:
        smoothing: Smoothing factor (default: 0.1)
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        """
        Args:
            pred: Model predictions (logits) of shape (N, C)
            target: Ground truth labels of shape (N,)
        
        Returns:
            Label smoothed cross entropy loss
        """
        n_class = pred.size(1)
        log_probs = F.log_softmax(pred, dim=1)
        
        # Create smooth labels
        with torch.no_grad():
            smooth_target = torch.zeros_like(pred)
            smooth_target.fill_(self.smoothing / (n_class - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        
        loss = F.kl_div(log_probs, smooth_target, reduction='batchmean')
        return loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for emotion and AU prediction.
    
    Combines emotion classification loss and AU detection loss with learnable weights.
    
    Args:
        num_emotions: Number of emotion classes (default: 14)
        num_aus: Number of action units (default: 18)
        initial_alpha: Initial weight for emotion loss (default: 0.7)
        learnable_weights: Whether to learn task weights (default: True)
    """
    def __init__(self, num_emotions=14, num_aus=18, initial_alpha=0.7, learnable_weights=True):
        super(MultiTaskLoss, self).__init__()
        self.num_emotions = num_emotions
        self.num_aus = num_aus
        
        if learnable_weights:
            # Learnable weights for each task
            self.log_vars = nn.Parameter(torch.zeros(2))
        else:
            self.register_buffer('log_vars', torch.tensor([0.0, 0.0]))
            self.log_vars[0] = torch.log(torch.tensor(initial_alpha))
            self.log_vars[1] = torch.log(torch.tensor(1 - initial_alpha))
    
    def forward(self, emotion_pred, emotion_true, au_pred, au_true):
        """
        Args:
            emotion_pred: Emotion predictions (logits) of shape (N, num_emotions)
            emotion_true: Emotion labels of shape (N,)
            au_pred: AU predictions (logits) of shape (N, num_aus)
            au_true: AU labels (multi-hot) of shape (N, num_aus)
        
        Returns:
            Total loss, emotion loss, AU loss
        """
        # Emotion loss (cross-entropy)
        emotion_loss = F.cross_entropy(emotion_pred, emotion_true)
        
        # AU loss (binary cross-entropy for multi-label)
        au_loss = F.binary_cross_entropy_with_logits(au_pred, au_true.float())
        
        # Homoscedastic uncertainty weighting
        # Higher uncertainty (larger sigma) -> lower weight
        precision_emotion = torch.exp(-self.log_vars[0])
        precision_au = torch.exp(-self.log_vars[1])
        
        total_loss = precision_emotion * emotion_loss + precision_au * au_loss
        
        # Add regularization term to prevent weights from going to infinity
        total_loss += 0.5 * (self.log_vars[0] + self.log_vars[1])
        
        return total_loss, emotion_loss, au_loss
    
    def get_task_weights(self):
        """Get the current task weights."""
        weights = F.softmax(self.log_vars, dim=0)
        return weights.detach().cpu().numpy()


class CombinedLoss(nn.Module):
    """
    Combined loss using Focal Loss and Label Smoothing.
    
    Args:
        alpha: Weighting factor for focal loss (default: 0.7)
        focal_gamma: Gamma for focal loss (default: 2.0)
        label_smoothing: Smoothing factor (default: 0.1)
    """
    def __init__(self, alpha=0.7, focal_gamma=2.0, label_smoothing=0.1):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.label_smoothing = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits) of shape (N, C)
            targets: Ground truth labels of shape (N,)
        
        Returns:
            Combined loss value
        """
        focal = self.focal_loss(inputs, targets)
        smoothed = self.label_smoothing(inputs, targets)
        return self.alpha * focal + (1 - self.alpha) * smoothed


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-label classification (AU detection).
    
    Args:
        smooth: Smoothing factor to avoid division by zero (default: 1.0)
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits) of shape (N, C)
            targets: Ground truth labels (multi-hot) of shape (N, C)
        
        Returns:
            Dice loss value
        """
        # Apply sigmoid to get probabilities
        inputs = torch.sigmoid(inputs)
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class AUCombinedLoss(nn.Module):
    """
    Combined loss for AU prediction (BCE + Dice).
    
    Args:
        bce_weight: Weight for BCE loss (default: 0.5)
        dice_weight: Weight for Dice loss (default: 0.5)
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(AUCombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss()
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits) of shape (N, C)
            targets: Ground truth labels (multi-hot) of shape (N, C)
        
        Returns:
            Combined AU loss
        """
        bce = F.binary_cross_entropy_with_logits(inputs, targets.float())
        dice = self.dice_loss(inputs, targets)
        return self.bce_weight * bce + self.dice_weight * dice


class AdaptiveLoss(nn.Module):
    """
    Adaptive loss that adjusts based on training progress.
    
    Starts with label smoothing and transitions to focal loss as training progresses.
    
    Args:
        total_epochs: Total number of training epochs (default: 100)
        start_smoothing: Initial label smoothing (default: 0.2)
        end_smoothing: Final label smoothing (default: 0.0)
        focal_gamma: Gamma for focal loss (default: 2.0)
    """
    def __init__(self, total_epochs=100, start_smoothing=0.2, end_smoothing=0.0, focal_gamma=2.0):
        super(AdaptiveLoss, self).__init__()
        self.total_epochs = total_epochs
        self.start_smoothing = start_smoothing
        self.end_smoothing = end_smoothing
        self.focal_gamma = focal_gamma
        self.current_epoch = 0
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits) of shape (N, C)
            targets: Ground truth labels of shape (N,)
        
        Returns:
            Adaptive loss value
        """
        # Calculate current smoothing factor
        progress = min(self.current_epoch / self.total_epochs, 1.0)
        current_smoothing = self.start_smoothing * (1 - progress) + self.end_smoothing * progress
        
        # Use label smoothing early, focal loss later
        if current_smoothing > 0.05:
            loss_fn = LabelSmoothingCrossEntropy(smoothing=current_smoothing)
        else:
            loss_fn = FocalLoss(gamma=self.focal_gamma)
        
        return loss_fn(inputs, targets)
    
    def step(self):
        """Advance to next epoch."""
        self.current_epoch += 1


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    # Create dummy data
    batch_size = 8
    num_emotions = 14
    num_aus = 18
    
    emotion_pred = torch.randn(batch_size, num_emotions)
    emotion_true = torch.randint(0, num_emotions, (batch_size,))
    au_pred = torch.randn(batch_size, num_aus)
    au_true = torch.randint(0, 2, (batch_size, num_aus)).float()
    
    # Test Focal Loss
    focal_loss = FocalLoss(gamma=2.0)
    loss = focal_loss(emotion_pred, emotion_true)
    print(f"Focal Loss: {loss.item():.4f}")
    
    # Test Label Smoothing
    ls_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss = ls_loss(emotion_pred, emotion_true)
    print(f"Label Smoothing Loss: {loss.item():.4f}")
    
    # Test Multi-Task Loss
    mt_loss = MultiTaskLoss(num_emotions, num_aus)
    total_loss, emo_loss, au_loss_val = mt_loss(emotion_pred, emotion_true, au_pred, au_true)
    print(f"Multi-Task Loss: {total_loss.item():.4f} (Emo: {emo_loss.item():.4f}, AU: {au_loss_val.item():.4f})")
    print(f"Task weights: {mt_loss.get_task_weights()}")
    
    # Test Combined Loss
    combined_loss = CombinedLoss(alpha=0.7, focal_gamma=2.0, label_smoothing=0.1)
    loss = combined_loss(emotion_pred, emotion_true)
    print(f"Combined Loss: {loss.item():.4f}")
    
    # Test Dice Loss
    dice_loss = DiceLoss()
    loss = dice_loss(au_pred, au_true)
    print(f"Dice Loss: {loss.item():.4f}")
    
    # Test AU Combined Loss
    au_combined = AUCombinedLoss()
    loss = au_combined(au_pred, au_true)
    print(f"AU Combined Loss: {loss.item():.4f}")
    
    # Test Adaptive Loss
    adaptive_loss = AdaptiveLoss(total_epochs=100)
    for epoch in [0, 25, 50, 75, 99]:
        adaptive_loss.current_epoch = epoch
        loss = adaptive_loss(emotion_pred, emotion_true)
        print(f"Adaptive Loss (epoch {epoch}): {loss.item():.4f}")
    
    print("All loss functions tested successfully!")
