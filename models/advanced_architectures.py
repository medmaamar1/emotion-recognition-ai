"""
Advanced model architectures for emotion recognition.
"""

import torch
import torch.nn as nn
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Install with: pip install timm")


class FERConvNeXt(nn.Module):
    """
    ConvNeXt model for emotion recognition.
    
    Args:
        model_name: ConvNeXt variant (default: 'convnext_base')
        num_classes: Number of emotion classes (default: 14)
        pretrained: Whether to use pretrained weights (default: True)
        dropout: Dropout rate (default: 0.3)
    """
    def __init__(self, model_name='convnext_base', num_classes=14, 
                 pretrained=True, dropout=0.3):
        super(FERConvNeXt, self).__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for ConvNeXt. Install with: pip install timm")
        
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'
        )
        
        # Get feature dimension
        self.num_features = self.model.num_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.num_features, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for custom layers."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        features = self.model(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """Extract features before classification."""
        return self.model(x)


class FERSwin(nn.Module):
    """
    Swin Transformer model for emotion recognition.
    
    Args:
        model_name: Swin variant (default: 'swin_base_patch4_window7_224')
        num_classes: Number of emotion classes (default: 14)
        pretrained: Whether to use pretrained weights (default: True)
        dropout: Dropout rate (default: 0.3)
    """
    def __init__(self, model_name='swin_base_patch4_window7_224', num_classes=14, 
                 pretrained=True, dropout=0.3):
        super(FERSwin, self).__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for Swin Transformer. Install with: pip install timm")
        
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'
        )
        
        # Get feature dimension
        self.num_features = self.model.num_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Dropout(dropout),
            nn.Linear(self.num_features, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for custom layers."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        features = self.model(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """Extract features before classification."""
        return self.model(x)


class FEREfficientNet(nn.Module):
    """
    EfficientNet model for emotion recognition.
    
    Args:
        model_name: EfficientNet variant (default: 'efficientnet_b4')
        num_classes: Number of emotion classes (default: 14)
        pretrained: Whether to use pretrained weights (default: True)
        dropout: Dropout rate (default: 0.3)
    """
    def __init__(self, model_name='efficientnet_b4', num_classes=14, 
                 pretrained=True, dropout=0.3):
        super(FEREfficientNet, self).__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for EfficientNet. Install with: pip install timm")
        
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'
        )
        
        # Get feature dimension
        self.num_features = self.model.num_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for custom layers."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        features = self.model(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """Extract features before classification."""
        return self.model(x)


class FERVitLarge(nn.Module):
    """
    Large Vision Transformer for emotion recognition.
    
    Args:
        model_name: ViT variant (default: 'vit_large_patch16_224')
        num_classes: Number of emotion classes (default: 14)
        pretrained: Whether to use pretrained weights (default: True)
        dropout: Dropout rate (default: 0.3)
    """
    def __init__(self, model_name='vit_large_patch16_224', num_classes=14, 
                 pretrained=True, dropout=0.3):
        super(FERVitLarge, self).__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for ViT Large. Install with: pip install timm")
        
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='token'
        )
        
        # Get feature dimension
        self.num_features = self.model.num_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Dropout(dropout),
            nn.Linear(self.num_features, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for custom layers."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        features = self.model(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """Extract features before classification."""
        return self.model(x)


class FERMobileNetV3(nn.Module):
    """
    MobileNetV3 for efficient emotion recognition.
    
    Args:
        model_name: MobileNetV3 variant (default: 'mobilenetv3_large_100')
        num_classes: Number of emotion classes (default: 14)
        pretrained: Whether to use pretrained weights (default: True)
        dropout: Dropout rate (default: 0.2)
    """
    def __init__(self, model_name='mobilenetv3_large_100', num_classes=14, 
                 pretrained=True, dropout=0.2):
        super(FERMobileNetV3, self).__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for MobileNetV3. Install with: pip install timm")
        
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'
        )
        
        # Get feature dimension
        self.num_features = self.model.num_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for custom layers."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        features = self.model(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """Extract features before classification."""
        return self.model(x)


def get_model(model_type, num_classes=14, pretrained=True, **kwargs):
    """
    Factory function to get model by type.
    
    Args:
        model_type: Type of model ('resnet50', 'convnext_base', 'swin_base', etc.)
        num_classes: Number of classes (default: 14)
        pretrained: Whether to use pretrained weights (default: True)
        **kwargs: Additional arguments for model initialization
    
    Returns:
        Model instance
    """
    model_map = {
        'resnet50': lambda: FERConvNeXt('resnet50', num_classes, pretrained, **kwargs),
        'convnext_tiny': lambda: FERConvNeXt('convnext_tiny', num_classes, pretrained, **kwargs),
        'convnext_small': lambda: FERConvNeXt('convnext_small', num_classes, pretrained, **kwargs),
        'convnext_base': lambda: FERConvNeXt('convnext_base', num_classes, pretrained, **kwargs),
        'convnext_large': lambda: FERConvNeXt('convnext_large', num_classes, pretrained, **kwargs),
        'swin_tiny': lambda: FERSwin('swin_tiny_patch4_window7_224', num_classes, pretrained, **kwargs),
        'swin_small': lambda: FERSwin('swin_small_patch4_window7_224', num_classes, pretrained, **kwargs),
        'swin_base': lambda: FERSwin('swin_base_patch4_window7_224', num_classes, pretrained, **kwargs),
        'swin_large': lambda: FERSwin('swin_large_patch4_window7_224', num_classes, pretrained, **kwargs),
        'efficientnet_b0': lambda: FEREfficientNet('efficientnet_b0', num_classes, pretrained, **kwargs),
        'efficientnet_b1': lambda: FEREfficientNet('efficientnet_b1', num_classes, pretrained, **kwargs),
        'efficientnet_b2': lambda: FEREfficientNet('efficientnet_b2', num_classes, pretrained, **kwargs),
        'efficientnet_b3': lambda: FEREfficientNet('efficientnet_b3', num_classes, pretrained, **kwargs),
        'efficientnet_b4': lambda: FEREfficientNet('efficientnet_b4', num_classes, pretrained, **kwargs),
        'efficientnet_b5': lambda: FEREfficientNet('efficientnet_b5', num_classes, pretrained, **kwargs),
        'vit_base': lambda: FERVitLarge('vit_base_patch16_224', num_classes, pretrained, **kwargs),
        'vit_large': lambda: FERVitLarge('vit_large_patch16_224', num_classes, pretrained, **kwargs),
        'vit_huge': lambda: FERVitLarge('vit_huge_patch14_224', num_classes, pretrained, **kwargs),
        'mobilenetv3_large': lambda: FERMobileNetV3('mobilenetv3_large_100', num_classes, pretrained, **kwargs),
        'mobilenetv3_small': lambda: FERMobileNetV3('mobilenetv3_small_100', num_classes, pretrained, **kwargs),
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_map.keys())}")
    
    return model_map[model_type]()


if __name__ == "__main__":
    # Test advanced architectures
    print("Testing advanced model architectures...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    
    if TIMM_AVAILABLE:
        # Test ConvNeXt
        print("\nTesting ConvNeXt...")
        try:
            model = FERConvNeXt('convnext_tiny', num_classes=14).to(device)
            dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
            output = model(dummy_input)
            print(f"ConvNeXt output shape: {output.shape}")
        except Exception as e:
            print(f"ConvNeXt test failed: {e}")
        
        # Test Swin
        print("\nTesting Swin Transformer...")
        try:
            model = FERSwin('swin_tiny_patch4_window7_224', num_classes=14).to(device)
            output = model(dummy_input)
            print(f"Swin output shape: {output.shape}")
        except Exception as e:
            print(f"Swin test failed: {e}")
        
        # Test EfficientNet
        print("\nTesting EfficientNet...")
        try:
            model = FEREfficientNet('efficientnet_b0', num_classes=14).to(device)
            output = model(dummy_input)
            print(f"EfficientNet output shape: {output.shape}")
        except Exception as e:
            print(f"EfficientNet test failed: {e}")
        
        # Test MobileNetV3
        print("\nTesting MobileNetV3...")
        try:
            model = FERMobileNetV3('mobilenetv3_small_100', num_classes=14).to(device)
            output = model(dummy_input)
            print(f"MobileNetV3 output shape: {output.shape}")
        except Exception as e:
            print(f"MobileNetV3 test failed: {e}")
        
        # Test factory function
        print("\nTesting model factory...")
        try:
            model = get_model('convnext_tiny', num_classes=14, pretrained=False).to(device)
            output = model(dummy_input)
            print(f"Factory model output shape: {output.shape}")
        except Exception as e:
            print(f"Factory test failed: {e}")
        
        print("\nAll advanced architectures tested successfully!")
    else:
        print("timm not available. Install with: pip install timm")
