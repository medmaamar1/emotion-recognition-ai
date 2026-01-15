"""
Multi-task learning model for emotion and AU prediction.
"""

import torch
import torch.nn as nn
from torchvision import models
from transformers import ViTForImageClassification, ViTConfig


class FERMultiTask(nn.Module):
    """
    Multi-task model for emotion classification and AU detection.
    
    Args:
        backbone: Backbone architecture ('resnet50', 'resnet101', 'vit')
        num_emotions: Number of emotion classes (default: 14)
        num_aus: Number of action units (default: 18)
        pretrained: Whether to use pretrained weights (default: True)
        dropout: Dropout rate (default: 0.3)
    """
    def __init__(self, backbone='resnet50', num_emotions=14, num_aus=18, 
                 pretrained=True, dropout=0.3):
        super(FERMultiTask, self).__init__()
        
        self.backbone_name = backbone
        self.num_emotions = num_emotions
        self.num_aus = num_aus
        
        # Shared backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if pretrained else None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == 'vit':
            if pretrained:
                self.backbone = ViTForImageClassification.from_pretrained(
                    'google/vit-base-patch16-224',
                    num_labels=num_emotions,
                    ignore_mismatched_sizes=True
                )
            else:
                config = ViTConfig(num_labels=num_emotions)
                self.backbone = ViTForImageClassification(config)
            in_features = 768  # ViT-base hidden size
            
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Emotion classification head
        self.emotion_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_emotions)
        )
        
        # AU detection head (multi-label classification)
        self.au_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_aus)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for new layers."""
        for m in self.shared_features.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        for m in self.emotion_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        for m in self.au_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images of shape (N, 3, H, W)
        
        Returns:
            Tuple of (emotion_logits, au_logits)
        """
        # Extract features from backbone
        if self.backbone_name == 'vit':
            features = self.backbone.vit(x)
            features = features.last_hidden_state[:, 0]  # [CLS] token
        else:
            features = self.backbone(x)
        
        # Shared feature extraction
        shared = self.shared_features(features)
        
        # Task-specific heads
        emotion_logits = self.emotion_head(shared)
        au_logits = self.au_head(shared)
        
        return emotion_logits, au_logits
    
    def get_features(self, x):
        """Extract shared features for visualization or other tasks."""
        if self.backbone_name == 'vit':
            features = self.backbone.vit(x)
            features = features.last_hidden_state[:, 0]
        else:
            features = self.backbone(x)
        
        return self.shared_features(features)


class FERMultiTaskWithAUAttention(nn.Module):
    """
    Multi-task model with AU attention mechanism.
    
    Uses attention to focus on AU-relevant regions.
    
    Args:
        backbone: Backbone architecture (default: 'resnet50')
        num_emotions: Number of emotion classes (default: 14)
        num_aus: Number of action units (default: 18)
        pretrained: Whether to use pretrained weights (default: True)
        dropout: Dropout rate (default: 0.3)
    """
    def __init__(self, backbone='resnet50', num_emotions=14, num_aus=18, 
                 pretrained=True, dropout=0.3):
        super(FERMultiTaskWithAUAttention, self).__init__()
        
        self.backbone_name = backbone
        self.num_emotions = num_emotions
        self.num_aus = num_aus
        
        # Shared backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            

        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # AU attention mechanism - learns which features are important for AU prediction
        self.au_attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_aus),
            nn.Sigmoid()
        )
        
        # AU context projection - projects AU attention back to feature space
        self.au_context = nn.Sequential(
            nn.Linear(num_aus, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512)
        )
        
        # Emotion classification head
        self.emotion_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_emotions)
        )
        
        # AU detection head
        self.au_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_aus)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images of shape (N, 3, H, W)
        
        Returns:
            Tuple of (emotion_logits, au_logits, au_attention_weights)
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Shared feature extraction
        shared = self.shared_features(features)
        
        # AU attention - learns which AUs are active
        au_attention = self.au_attention(shared)  # (batch, num_aus)
        
        # Project AU attention back to feature space to create context
        au_context = self.au_context(au_attention)  # (batch, 512)
        
        # Apply AU context to shared features for AU prediction
        # This uses the AU context to modulate the shared features
        attended_features = shared * torch.sigmoid(au_context)
        
        # Task-specific heads
        emotion_logits = self.emotion_head(shared)
        au_logits = self.au_head(attended_features)
        
        return emotion_logits, au_logits, au_attention


class FERMultiTaskWithCrossAttention(nn.Module):
    """
    Multi-task model with cross-attention between emotion and AU tasks.
    
    Args:
        backbone: Backbone architecture (default: 'resnet50')
        num_emotions: Number of emotion classes (default: 14)
        num_aus: Number of action units (default: 18)
        pretrained: Whether to use pretrained weights (default: True)
        dropout: Dropout rate (default: 0.3)
    """
    def __init__(self, backbone='resnet50', num_emotions=14, num_aus=18, 
                 pretrained=True, dropout=0.3):
        super(FERMultiTaskWithCrossAttention, self).__init__()
        
        self.backbone_name = backbone
        self.num_emotions = num_emotions
        self.num_aus = num_aus
        
        # Shared backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            

        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific embeddings
        self.emotion_embedding = nn.Parameter(torch.randn(1, 1, 512))
        self.au_embedding = nn.Parameter(torch.randn(1, 1, 512))
        
        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(512, num_heads=8, dropout=dropout)
        
        # Emotion classification head
        self.emotion_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_emotions)
        )
        
        # AU detection head
        self.au_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_aus)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images of shape (N, 3, H, W)
        
        Returns:
            Tuple of (emotion_logits, au_logits)
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Shared feature extraction
        shared = self.shared_features(features)  # (N, 512)
        
        # Reshape for attention
        shared = shared.unsqueeze(0)  # (1, N, 512)
        
        # Expand task embeddings
        batch_size = shared.size(1)
        emotion_emb = self.emotion_embedding.expand(1, batch_size, -1)
        au_emb = self.au_embedding.expand(1, batch_size, -1)
        
        # Cross-attention: emotion features attend to AU features
        emotion_attended, _ = self.cross_attention(emotion_emb, shared, shared)
        emotion_attended = emotion_attended.squeeze(0)  # (N, 512)
        
        # Cross-attention: AU features attend to emotion features
        au_attended, _ = self.cross_attention(au_emb, shared, shared)
        au_attended = au_attended.squeeze(0)  # (N, 512)
        
        # Task-specific heads
        emotion_logits = self.emotion_head(emotion_attended)
        au_logits = self.au_head(au_attended)
        
        return emotion_logits, au_logits


if __name__ == "__main__":
    # Test multi-task models
    print("Testing multi-task models...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    
    # Test FERMultiTask
    print("\nTesting FERMultiTask...")
    model = FERMultiTask(backbone='resnet50', num_emotions=14, num_aus=18).to(device)
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    emotion_logits, au_logits = model(dummy_input)
    print(f"Emotion logits shape: {emotion_logits.shape}")
    print(f"AU logits shape: {au_logits.shape}")
    
    # Test FERMultiTaskWithAUAttention
    print("\nTesting FERMultiTaskWithAUAttention...")
    model = FERMultiTaskWithAUAttention(backbone='resnet50', num_emotions=14, num_aus=18).to(device)
    emotion_logits, au_logits, au_attention = model(dummy_input)
    print(f"Emotion logits shape: {emotion_logits.shape}")
    print(f"AU logits shape: {au_logits.shape}")
    print(f"AU attention shape: {au_attention.shape}")
    
    # Test FERMultiTaskWithCrossAttention
    print("\nTesting FERMultiTaskWithCrossAttention...")
    model = FERMultiTaskWithCrossAttention(backbone='resnet50', num_emotions=14, num_aus=18).to(device)
    emotion_logits, au_logits = model(dummy_input)
    print(f"Emotion logits shape: {emotion_logits.shape}")
    print(f"AU logits shape: {au_logits.shape}")
    
    # Test feature extraction
    print("\nTesting feature extraction...")
    features = model.get_features(dummy_input)
    print(f"Features shape: {features.shape}")
    
    print("\nAll multi-task models tested successfully!")
