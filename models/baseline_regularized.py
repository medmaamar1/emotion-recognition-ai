"""
Regularized baseline models for emotion recognition with dropout and other regularization techniques.
"""

import torch
import torch.nn as nn
from torchvision import models
from transformers import ViTForImageClassification, ViTConfig


class FERBaselineRegularized(nn.Module):
    """
    Regularized baseline model for emotion recognition with dropout and regularization.
    
    Args:
        model_type: Model architecture ('resnet50', 'efficientnet_b0', 'vit')
        num_classes: Number of emotion classes (default: 14)
        pretrained: Whether to use pretrained weights (default: True)
        dropout_rate: Dropout rate (default: 0.3)
        use_label_smoothing: Whether to use label smoothing (default: True)
    """
    def __init__(self, model_type='resnet50', num_classes=14, pretrained=True, 
                 dropout_rate=0.3, use_label_smoothing=True):
        super(FERBaselineRegularized, self).__init__()
        self.model_type = model_type
        self.dropout_rate = dropout_rate
        self.use_label_smoothing = use_label_smoothing
        
        if model_type == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            in_features = self.model.fc.in_features
            # Add dropout before final layer
            self.dropout = nn.Dropout(p=dropout_rate)
            self.model.fc = nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features // 2, num_classes)
            )
            
        elif model_type == 'efficientnet_b0':
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            in_features = self.model.classifier[1].in_features
            self.dropout = nn.Dropout(p=dropout_rate)
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features // 2, num_classes)
            )
            
        elif model_type == 'vit':
            # Using ViT-Base-Patch16-224
            if pretrained:
                self.model = ViTForImageClassification.from_pretrained(
                    'google/vit-base-patch16-224',
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True
                )
            else:
                config = ViTConfig(num_labels=num_classes)
                self.model = ViTForImageClassification(config)
            # ViT has built-in dropout, configure it
            if hasattr(self.model, 'vit'):
                self.model.vit.dropout = dropout_rate
                self.model.vit.attention_dropout = dropout_rate
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def forward(self, x):
        if self.model_type == 'resnet50':
            x = self.model(x)
            x = self.dropout(x)
            return self.model.fc(x)
        elif self.model_type == 'efficientnet_b0':
            x = self.model(x)
            x = self.dropout(x)
            return self.model.classifier(x)
        elif self.model_type == 'vit':
            outputs = self.model(x)
            return outputs.logits


class FERBaselineWithLabelSmoothing(nn.Module):
    """
    Baseline model with label smoothing for better generalization.
    """
    def __init__(self, model_type='resnet50', num_classes=14, pretrained=True,
                 smoothing=0.1, dropout_rate=0.3):
        super(FERBaselineWithLabelSmoothing, self).__init__()
        self.model_type = model_type
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.dropout_rate = dropout_rate
        
        if model_type == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            in_features = self.model.fc.in_features
            self.dropout = nn.Dropout(p=dropout_rate)
            self.model.fc = nn.Linear(in_features, num_classes)
            
        elif model_type == 'efficientnet_b0':
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            in_features = self.model.classifier[1].in_features
            self.dropout = nn.Dropout(p=dropout_rate)
            self.model.classifier = nn.Linear(in_features, num_classes)
            
        elif model_type == 'vit':
            if pretrained:
                self.model = ViTForImageClassification.from_pretrained(
                    'google/vit-base-patch16-224',
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True
                )
            else:
                config = ViTConfig(num_labels=num_classes)
                self.model = ViTForImageClassification(config)
            self.dropout = nn.Dropout(p=dropout_rate)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def forward(self, x):
        if self.model_type == 'resnet50':
            x = self.model(x)
            x = self.dropout(x)
            logits = self.model.fc(x)
            # Apply label smoothing
            return logits
        
        elif self.model_type == 'efficientnet_b0':
            x = self.model(x)
            x = self.dropout(x)
            logits = self.model.classifier(x)
            return logits
        
        elif self.model_type == 'vit':
            outputs = self.model(x)
            logits = outputs.logits
            return logits
    
    def get_label_smoothing_loss(self, logits, targets):
        """
        Apply label smoothing to cross-entropy loss.
        
        Args:
            logits: Model predictions (N, num_classes)
            targets: Ground truth labels (N,)
        
        Returns:
            Smoothed cross-entropy loss
        """
        log_probs = torch.log_softmax(logits, dim=1)
        
        # Create smooth labels
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        # Calculate loss
        loss = -torch.sum(smooth_targets * log_probs, dim=1).mean()
        return loss


if __name__ == "__main__":
    # Test regularized models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Testing Regularized ResNet50...")
    resnet_reg = FERBaselineRegularized(model_type='resnet50', dropout_rate=0.3).to(device)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = resnet_reg(dummy_input)
    print(f"Regularized ResNet50 Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in resnet_reg.parameters())}")
    
    print("\nTesting Label Smoothing...")
    resnet_ls = FERBaselineWithLabelSmoothing(model_type='resnet50', smoothing=0.1, dropout_rate=0.3).to(device)
    output = resnet_ls(dummy_input)
    print(f"Label Smoothing Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in resnet_ls.parameters())}")
