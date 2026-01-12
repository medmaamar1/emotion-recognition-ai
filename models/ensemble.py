"""
Ensemble methods for combining multiple models.
"""

import torch
import torch.nn as nn
import numpy as np


class FEREnsemble(nn.Module):
    """
    Simple ensemble that averages predictions from multiple models.
    
    Args:
        models: List of PyTorch models
        weights: Optional weights for each model (default: None for equal weights)
    """
    def __init__(self, models, weights=None):
        super(FEREnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = [1.0] * len(models)
        else:
            self.weights = weights
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        self.weights = torch.tensor(self.weights, dtype=torch.float32)
    
    def forward(self, x):
        """
        Forward pass through all models and average predictions.
        
        Args:
            x: Input images of shape (N, 3, H, W)
        
        Returns:
            Averaged predictions
        """
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = torch.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def predict_with_uncertainty(self, x):
        """
        Predict with uncertainty estimation.
        
        Args:
            x: Input images of shape (N, 3, H, W)
        
        Returns:
            Tuple of (mean_prediction, std_prediction)
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        return mean_pred, std_pred


class WeightedEnsemble(nn.Module):
    """
    Learnable weighted ensemble.
    
    Args:
        models: List of PyTorch models
        num_classes: Number of output classes
    """
    def __init__(self, models, num_classes=14):
        super(WeightedEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        
        # Learnable weights
        self.weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
    
    def forward(self, x):
        """
        Forward pass with learnable weights.
        
        Args:
            x: Input images of shape (N, 3, H, W)
        
        Returns:
            Weighted averaged predictions
        """
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Normalize weights
        norm_weights = torch.softmax(self.weights, dim=0)
        
        # Weighted average
        ensemble_pred = torch.zeros_like(predictions[0])
        for pred, weight in zip(predictions, norm_weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred


class VotingEnsemble(nn.Module):
    """
    Voting ensemble that uses majority voting.
    
    Args:
        models: List of PyTorch models
        voting_type: 'hard' or 'soft' (default: 'soft')
    """
    def __init__(self, models, voting_type='soft'):
        super(VotingEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.voting_type = voting_type
    
    def forward(self, x):
        """
        Forward pass with voting.
        
        Args:
            x: Input images of shape (N, 3, H, W)
        
        Returns:
            Voted predictions
        """
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        if self.voting_type == 'soft':
            # Soft voting: average probabilities
            probs = torch.stack([torch.softmax(p, dim=1) for p in predictions])
            ensemble_pred = torch.mean(probs, dim=0)
            return ensemble_pred
        else:
            # Hard voting: majority vote
            pred_classes = [torch.argmax(p, dim=1) for p in predictions]
            pred_classes = torch.stack(pred_classes, dim=1)
            
            # Majority vote
            ensemble_pred = torch.mode(pred_classes, dim=1)[0]
            
            # Convert back to logits (one-hot)
            num_classes = predictions[0].size(1)
            ensemble_pred_onehot = torch.zeros_like(predictions[0])
            for i, cls in enumerate(ensemble_pred):
                ensemble_pred_onehot[i, cls] = 1.0
            
            return ensemble_pred_onehot


class StackingEnsemble(nn.Module):
    """
    Stacking ensemble with a meta-learner.
    
    Args:
        models: List of PyTorch models (base learners)
        num_classes: Number of output classes
        hidden_dim: Hidden dimension for meta-learner (default: 256)
    """
    def __init__(self, models, num_classes=14, hidden_dim=256):
        super(StackingEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.num_classes = num_classes
        
        # Meta-learner
        self.meta_learner = nn.Sequential(
            nn.Linear(self.num_models * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass with stacking.
        
        Args:
            x: Input images of shape (N, 3, H, W)
        
        Returns:
            Meta-learner predictions
        """
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Concatenate predictions
        stacked = torch.cat(predictions, dim=1)
        
        # Pass through meta-learner
        ensemble_pred = self.meta_learner(stacked)
        
        return ensemble_pred


class BaggingEnsemble(nn.Module):
    """
    Bagging ensemble with bootstrap sampling.
    
    Args:
        models: List of PyTorch models trained on different bootstrap samples
        num_classes: Number of output classes
    """
    def __init__(self, models, num_classes=14):
        super(BaggingEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        Forward pass with bagging.
        
        Args:
            x: Input images of shape (N, 3, H, W)
        
        Returns:
            Averaged predictions
        """
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
        
        return ensemble_pred


class SnapshotEnsemble(nn.Module):
    """
    Snapshot ensemble using models from different training epochs.
    
    Args:
        models: List of PyTorch models (snapshots from different epochs)
        weights: Optional weights for each snapshot (default: None)
    """
    def __init__(self, models, weights=None):
        super(SnapshotEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            # Give more weight to later snapshots
            self.weights = torch.linspace(0.5, 1.0, len(models))
        else:
            self.weights = torch.tensor(weights, dtype=torch.float32)
        
        # Normalize weights
        self.weights = self.weights / self.weights.sum()
    
    def forward(self, x):
        """
        Forward pass with snapshot averaging.
        
        Args:
            x: Input images of shape (N, 3, H, W)
        
        Returns:
            Weighted averaged predictions
        """
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = torch.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred


class DiverseEnsemble(nn.Module):
    """
    Ensemble that encourages diversity among models.
    
    Args:
        models: List of PyTorch models
        num_classes: Number of output classes
        diversity_weight: Weight for diversity loss (default: 0.1)
    """
    def __init__(self, models, num_classes=14, diversity_weight=0.1):
        super(DiverseEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_classes = num_classes
        self.diversity_weight = diversity_weight
    
    def forward(self, x):
        """
        Forward pass with diversity-aware averaging.
        
        Args:
            x: Input images of shape (N, 3, H, W)
        
        Returns:
            Averaged predictions
        """
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
        
        return ensemble_pred
    
    def diversity_loss(self, x):
        """
        Compute diversity loss to encourage model diversity.
        
        Args:
            x: Input images of shape (N, 3, H, W)
        
        Returns:
            Diversity loss
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(torch.softmax(pred, dim=1))
        
        predictions = torch.stack(predictions)
        
        # Compute pairwise KL divergence
        diversity_loss = 0.0
        num_pairs = 0
        
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                kl_div = nn.functional.kl_div(
                    torch.log(predictions[i] + 1e-10),
                    predictions[j] + 1e-10,
                    reduction='batchmean'
                )
                diversity_loss += kl_div
                num_pairs += 1
        
        if num_pairs > 0:
            diversity_loss /= num_pairs
        
        return diversity_loss


def create_ensemble(model_configs, device='cuda'):
    """
    Factory function to create ensemble from model configurations.
    
    Args:
        model_configs: List of (model_class, model_args) tuples
        device: Device to load models on
    
    Returns:
        Ensemble model
    """
    models = []
    
    for model_class, model_args in model_configs:
        model = model_class(**model_args).to(device)
        models.append(model)
    
    return FEREnsemble(models)


if __name__ == "__main__":
    # Test ensemble methods
    print("Testing ensemble methods...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    num_classes = 14
    
    # Create dummy models
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, num_classes)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # Test FEREnsemble
    print("\nTesting FEREnsemble...")
    models = [DummyModel() for _ in range(3)]
    ensemble = FEREnsemble(models).to(device)
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    output = ensemble(dummy_input)
    print(f"Ensemble output shape: {output.shape}")
    
    # Test with uncertainty
    mean_pred, std_pred = ensemble.predict_with_uncertainty(dummy_input)
    print(f"Mean pred shape: {mean_pred.shape}, Std pred shape: {std_pred.shape}")
    
    # Test WeightedEnsemble
    print("\nTesting WeightedEnsemble...")
    weighted_ensemble = WeightedEnsemble(models, num_classes=num_classes).to(device)
    output = weighted_ensemble(dummy_input)
    print(f"Weighted ensemble output shape: {output.shape}")
    print(f"Learned weights: {torch.softmax(weighted_ensemble.weights, dim=0)}")
    
    # Test VotingEnsemble
    print("\nTesting VotingEnsemble...")
    voting_ensemble = VotingEnsemble(models, voting_type='soft').to(device)
    output = voting_ensemble(dummy_input)
    print(f"Voting ensemble output shape: {output.shape}")
    
    # Test StackingEnsemble
    print("\nTesting StackingEnsemble...")
    stacking_ensemble = StackingEnsemble(models, num_classes=num_classes).to(device)
    output = stacking_ensemble(dummy_input)
    print(f"Stacking ensemble output shape: {output.shape}")
    
    # Test SnapshotEnsemble
    print("\nTesting SnapshotEnsemble...")
    snapshot_ensemble = SnapshotEnsemble(models).to(device)
    output = snapshot_ensemble(dummy_input)
    print(f"Snapshot ensemble output shape: {output.shape}")
    print(f"Snapshot weights: {snapshot_ensemble.weights}")
    
    # Test DiverseEnsemble
    print("\nTesting DiverseEnsemble...")
    diverse_ensemble = DiverseEnsemble(models, num_classes=num_classes).to(device)
    output = diverse_ensemble(dummy_input)
    print(f"Diverse ensemble output shape: {output.shape}")
    
    # Test diversity loss
    div_loss = diverse_ensemble.diversity_loss(dummy_input)
    print(f"Diversity loss: {div_loss.item():.4f}")
    
    print("\nAll ensemble methods tested successfully!")
