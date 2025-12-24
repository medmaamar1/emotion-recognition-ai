import torch
import torch.nn as nn
from torchvision import models
from transformers import ViTForImageClassification, ViTConfig

class FERBaseline(nn.Module):
    def __init__(self, model_type='resnet50', num_classes=14, pretrained=True):
        super(FERBaseline, self).__init__()
        self.model_type = model_type
        
        if model_type == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
            
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
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, x):
        if self.model_type == 'resnet50':
            return self.model(x)
        elif self.model_type == 'vit':
            outputs = self.model(x)
            return outputs.logits

if __name__ == "__main__":
    # Test architectures
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Testing ResNet50...")
    resnet = FERBaseline(model_type='resnet50').to(device)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = resnet(dummy_input)
    print(f"ResNet Output shape: {output.shape}")
    
    print("\nTesting ViT...")
    vit = FERBaseline(model_type='vit').to(device)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = vit(dummy_input)
    print(f"ViT Output shape: {output.shape}")
