"""
Test-Time Augmentation (TTA) for improved inference accuracy.
"""

import torch
import numpy as np
from PIL import Image


class TTAInference:
    """
    Test-Time Augmentation wrapper for models.
    
    Applies multiple augmentations to the input image and averages predictions.
    
    Args:
        model: PyTorch model
        tta_transforms: List of augmentation transforms
        device: Device to run model on
        aggregation_method: Method to aggregate predictions ('mean', 'max', 'vote')
    """
    def __init__(self, model, tta_transforms, device, aggregation_method='mean'):
        self.model = model
        self.tta_transforms = tta_transforms
        self.device = device
        self.aggregation_method = aggregation_method
        
    def predict(self, image):
        """
        Apply TTA and return averaged prediction.
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            Averaged prediction
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for transform in self.tta_transforms:
                # Apply transform
                if isinstance(image, torch.Tensor):
                    # If already tensor, convert back for albumentations
                    img_np = image.cpu().numpy().transpose(1, 2, 0)
                    # Denormalize for albumentations
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_np = (image * std + mean).cpu().numpy().transpose(1, 2, 0)
                    img_np = (img_np * 255).astype('uint8')
                else:
                    img_np = image
                
                # Apply augmentation
                augmented = transform(image=img_np)['image']
                augmented = augmented.unsqueeze(0).to(self.device)
                
                # Get prediction
                pred = self.model(augmented)
                predictions.append(pred)
        
        # Aggregate predictions
        if self.aggregation_method == 'mean':
            ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
        elif self.aggregation_method == 'max':
            ensemble_pred = torch.max(torch.stack(predictions), dim=0)[0]
        elif self.aggregation_method == 'vote':
            # Soft voting
            probs = torch.softmax(torch.stack(predictions), dim=2)
            ensemble_pred = torch.mean(probs, dim=0)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        return ensemble_pred
    
    def predict_batch(self, images):
        """
        Apply TTA to a batch of images.
        
        Args:
            images: List of PIL Images or numpy arrays
        
        Returns:
            List of averaged predictions
        """
        predictions = []
        for image in images:
            pred = self.predict(image)
            predictions.append(pred)
        return torch.cat(predictions, dim=0)
    
    def predict_with_confidence(self, image):
        """
        Apply TTA and return prediction with confidence intervals.
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            Tuple of (mean_prediction, std_prediction)
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for transform in self.tta_transforms:
                if isinstance(image, torch.Tensor):
                    img_np = image.cpu().numpy().transpose(1, 2, 0)
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_np = (image * std + mean).cpu().numpy().transpose(1, 2, 0)
                    img_np = (img_np * 255).astype('uint8')
                else:
                    img_np = image
                
                augmented = transform(image=img_np)['image']
                augmented = augmented.unsqueeze(0).to(self.device)
                pred = self.model(augmented)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        return mean_pred, std_pred


class AdaptiveTTA:
    """
    Adaptive TTA that selects augmentations based on prediction confidence.
    
    Only applies additional augmentations if initial prediction is uncertain.
    
    Args:
        model: PyTorch model
        tta_transforms: List of augmentation transforms
        device: Device to run model on
        confidence_threshold: Threshold for stopping TTA (default: 0.9)
        max_augmentations: Maximum number of augmentations to apply (default: 5)
    """
    def __init__(self, model, tta_transforms, device, confidence_threshold=0.9, max_augmentations=5):
        self.model = model
        self.tta_transforms = tta_transforms
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.max_augmentations = max_augmentations
        
    def predict(self, image):
        """
        Apply adaptive TTA.
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            Prediction
        """
        self.model.eval()
        predictions = []
        confidences = []
        
        with torch.no_grad():
            # Apply first transform (usually no augmentation)
            transform = self.tta_transforms[0]
            if isinstance(image, torch.Tensor):
                img_np = image.cpu().numpy().transpose(1, 2, 0)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_np = (image * std + mean).cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * 255).astype('uint8')
            else:
                img_np = image
            
            augmented = transform(image=img_np)['image']
            augmented = augmented.unsqueeze(0).to(self.device)
            pred = self.model(augmented)
            predictions.append(pred)
            
            # Check confidence
            probs = torch.softmax(pred, dim=1)
            max_prob, _ = torch.max(probs, dim=1)
            confidences.append(max_prob.item())
            
            # If confident, return early
            if max_prob.item() >= self.confidence_threshold:
                return pred
            
            # Otherwise, apply more augmentations
            num_augmentations = min(len(self.tta_transforms) - 1, self.max_augmentations - 1)
            for i in range(1, num_augmentations + 1):
                transform = self.tta_transforms[i]
                augmented = transform(image=img_np)['image']
                augmented = augmented.unsqueeze(0).to(self.device)
                pred = self.model(augmented)
                predictions.append(pred)
                
                probs = torch.softmax(pred, dim=1)
                max_prob, _ = torch.max(probs, dim=1)
                confidences.append(max_prob.item())
        
        # Average predictions
        ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
        
        return ensemble_pred


def get_tta_transforms(img_size=224, mode='standard'):
    """
    Get TTA transforms for different modes.
    
    Args:
        img_size: Target image size (default: 224)
        mode: 'standard', 'extended', or 'minimal' (default: 'standard')
    
    Returns:
        List of TTA transforms
    """
    from utils.augmentation import get_val_transform
    
    base_transform = get_val_transform(img_size)
    
    if mode == 'minimal':
        # Only horizontal flip
        h_flip = get_val_transform(img_size)
        h_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ])
        return [base_transform, h_transform]
    
    elif mode == 'standard':
        # Standard TTA: original, h-flip, v-flip, rotate ±15
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        h_flip = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ])
        
        v_flip = A.Compose([
            A.Resize(img_size, img_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ])
        
        rotate_15 = A.Compose([
            A.Resize(img_size, img_size),
            A.Rotate(limit=15, border_mode=0, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ])
        
        rotate_neg_15 = A.Compose([
            A.Resize(img_size, img_size),
            A.Rotate(limit=-15, border_mode=0, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ])
        
        return [base_transform, h_flip, v_flip, rotate_15, rotate_neg_15]
    
    elif mode == 'extended':
        # Extended TTA with more augmentations
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        transforms = [base_transform]
        
        # Flips
        for flip_type, p in [('h', 1.0), ('v', 1.0)]:
            if flip_type == 'h':
                flip = A.HorizontalFlip(p=p)
            else:
                flip = A.VerticalFlip(p=p)
            
            transforms.append(A.Compose([
                A.Resize(img_size, img_size),
                flip,
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                ),
                ToTensorV2(),
            ]))
        
        # Rotations
        for angle in [5, 10, 15, -5, -10, -15]:
            transforms.append(A.Compose([
                A.Resize(img_size, img_size),
                A.Rotate(limit=angle, border_mode=0, p=1.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                ),
                ToTensorV2(),
            ]))
        
        # Color jitter
        transforms.append(A.Compose([
            A.Resize(img_size, img_size),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]))
        
        return transforms
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    # Test TTA
    import torch.nn as nn
    
    print("Testing Test-Time Augmentation...")
    
    # Create a simple model
    class DummyModel(nn.Module):
        def __init__(self, num_classes=14):
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
    
    model = DummyModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Get TTA transforms
    from utils.augmentation import get_val_transform
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    base_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])
    
    h_flip = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])
    
    tta_transforms = [base_transform, h_flip]
    
    # Test TTAInference
    print("\nTesting TTAInference...")
    tta = TTAInference(model, tta_transforms, device)
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    pred = tta.predict(dummy_image)
    print(f"Prediction shape: {pred.shape}")
    
    # Test with confidence
    mean_pred, std_pred = tta.predict_with_confidence(dummy_image)
    print(f"Mean pred shape: {mean_pred.shape}, Std pred shape: {std_pred.shape}")
    
    # Test AdaptiveTTA
    print("\nTesting AdaptiveTTA...")
    adaptive_tta = AdaptiveTTA(model, tta_transforms, device, confidence_threshold=0.95)
    pred = adaptive_tta.predict(dummy_image)
    print(f"Adaptive prediction shape: {pred.shape}")
    
    print("\nAll TTA utilities tested successfully!")
