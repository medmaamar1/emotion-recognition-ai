"""
Advanced data augmentation using Albumentations for emotion recognition.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch


def get_train_transform(img_size=224):
    """
    Get advanced training augmentation pipeline using Albumentations.
    
    Args:
        img_size: Target image size (default: 224)
    
    Returns:
        Albumentations compose object for training
    """
    return A.Compose([
        # Resize with padding to maintain aspect ratio
        A.Resize(img_size + 32, img_size + 32),
        
        # Random crop to target size
        A.RandomCrop(img_size, img_size, p=1.0),
        
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Rotate(limit=30, border_mode=0, p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.1, 
            rotate_limit=0, 
            border_mode=0, 
            p=0.5
        ),
        
        # Color and brightness augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
        ], p=0.5),
        
        # Noise and blur augmentations
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.3),
        
        # Distortion augmentations
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=1.0),
        ], p=0.2),
        
        # Coarse dropout (simulates occlusion)
        A.CoarseDropout(
            max_holes=8,
            max_height=int(img_size * 0.15),
            max_width=int(img_size * 0.15),
            min_holes=1,
            min_height=int(img_size * 0.05),
            min_width=int(img_size * 0.05),
            fill_value=0,
            p=0.3
        ),
        
        # Cutout (random square erasing)
        A.Cutout(
            num_holes=1,
            max_h_size=int(img_size * 0.2),
            max_w_size=int(img_size * 0.2),
            fill_value=0,
            p=0.2
        ),
        
        # Normalization (ImageNet stats)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        
        # Convert to tensor
        ToTensorV2(),
    ])


def get_val_transform(img_size=224):
    """
    Get validation/test augmentation pipeline.
    
    Args:
        img_size: Target image size (default: 224)
    
    Returns:
        Albumentations compose object for validation
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size=224):
    """
    Get test-time augmentation transforms for inference.
    
    Args:
        img_size: Target image size (default: 224)
    
    Returns:
        List of Albumentations compose objects for TTA
    """
    base_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])
    
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


def apply_tta(model, image, tta_transforms, device):
    """
    Apply test-time augmentation and average predictions.
    
    Args:
        model: PyTorch model
        image: PIL Image or numpy array
        tta_transforms: List of TTA transforms
        device: Device to run model on
    
    Returns:
        Averaged predictions
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for transform in tta_transforms:
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
            augmented = augmented.unsqueeze(0).to(device)
            
            # Get prediction
            pred = model(augmented)
            predictions.append(pred)
    
    # Average predictions
    ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
    
    return ensemble_pred


def get_simple_augmentation(img_size=224):
    """
    Get simple augmentation pipeline for quick experiments.
    
    Args:
        img_size: Target image size (default: 224)
    
    Returns:
        Albumentations compose object for simple augmentation
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])


def get_strong_augmentation(img_size=224):
    """
    Get strong augmentation pipeline for challenging cases.
    
    Args:
        img_size: Target image size (default: 224)
    
    Returns:
        Albumentations compose object for strong augmentation
    """
    return A.Compose([
        A.Resize(img_size + 64, img_size + 64),
        A.RandomCrop(img_size, img_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=45, border_mode=0, p=0.7),
        A.ShiftScaleRotate(
            shift_limit=0.15,
            scale_limit=0.15,
            rotate_limit=0,
            border_mode=0,
            p=0.6
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=1.0),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.7),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 70.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 9), p=1.0),
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.MedianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.4),
        A.OneOf([
            A.ElasticTransform(alpha=1.5, sigma=60, alpha_affine=60, p=1.0),
            A.GridDistortion(num_steps=7, distort_limit=0.4, p=1.0),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=1.0),
        ], p=0.3),
        A.CoarseDropout(
            max_holes=12,
            max_height=int(img_size * 0.2),
            max_width=int(img_size * 0.2),
            min_holes=1,
            min_height=int(img_size * 0.05),
            min_width=int(img_size * 0.05),
            fill_value=0,
            p=0.4
        ),
        A.Cutout(
            num_holes=2,
            max_h_size=int(img_size * 0.25),
            max_w_size=int(img_size * 0.25),
            fill_value=0,
            p=0.3
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])


if __name__ == "__main__":
    # Test the augmentation pipelines
    import numpy as np
    from PIL import Image
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    print("Testing augmentation pipelines...")
    
    # Test train transform
    train_transform = get_train_transform()
    augmented = train_transform(image=dummy_image)
    print(f"Train transform output shape: {augmented['image'].shape}")
    
    # Test val transform
    val_transform = get_val_transform()
    augmented = val_transform(image=dummy_image)
    print(f"Val transform output shape: {augmented['image'].shape}")
    
    # Test TTA transforms
    tta_transforms = get_tta_transforms()
    print(f"Number of TTA transforms: {len(tta_transforms)}")
    
    print("All augmentation pipelines tested successfully!")
