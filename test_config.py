#!/usr/bin/env python3
"""
Test script to verify configuration works correctly on both local and Kaggle environments.
"""

import os
import sys

# Test configuration loading
print("=" * 60)
print("Testing Configuration")
print("=" * 60)

try:
    from config import get_config
    config = get_config()
    print("✓ Config module loaded successfully")
except Exception as e:
    print(f"✗ Error loading config: {e}")
    sys.exit(1)

print(f"\nRunning on Kaggle: {config['is_kaggle']}")
print(f"\nPaths:")
print(f"  - Data Root (Aligned): {config['data_root_aligned']}")
print(f"  - Data Root (Raw): {config['data_root_raw']}")
print(f"  - Partition File: {config['partition_file']}")
print(f"  - Emotion File: {config['emotion_file']}")
print(f"  - AU File: {config['au_file']}")
print(f"  - Output Directory: {config['output_dir']}")
print(f"  - Checkpoint Directory: {config['checkpoint_dir']}")
print(f"  - Log Directory: {config['log_dir']}")

# Test path existence
print("\n" + "=" * 60)
print("Path Verification")
print("=" * 60)

paths_to_check = {
    'Aligned Images': config['data_root_aligned'],
    'Raw Images': config['data_root_raw'],
    'Partition File': config['partition_file'],
    'Emotion File': config['emotion_file'],
    'AU File': config['au_file'],
}

all_exist = True
for name, path in paths_to_check.items():
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"{status} {name}: {path}")
    if not exists:
        all_exist = False

# Test output directories
print("\n" + "=" * 60)
print("Output Directories")
print("=" * 60)

output_dirs = {
    'Output': config['output_dir'],
    'Checkpoint': config['checkpoint_dir'],
    'Log': config['log_dir']
}

for name, path in output_dirs.items():
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"{status} {name}: {path}")
    if not exists:
        try:
            os.makedirs(path, exist_ok=True)
            print(f"  └─ Created directory")
        except Exception as e:
            print(f"  └─ Error creating directory: {e}")
            all_exist = False

# Test dataset loading
print("\n" + "=" * 60)
print("Dataset Loading Test")
print("=" * 60)

try:
    from scripts.dataset import RAFCEDataset
    
    # Test training dataset
    print("\nLoading training dataset (partition_id=0)...")
    train_dataset = RAFCEDataset(partition_id=0, use_aligned=True)
    print(f"✓ Training dataset loaded: {len(train_dataset)} samples")
    
    # Test validation dataset
    print("\nLoading validation dataset (partition_id=2)...")
    val_dataset = RAFCEDataset(partition_id=2, use_aligned=True)
    print(f"✓ Validation dataset loaded: {len(val_dataset)} samples")
    
    # Test test dataset
    print("\nLoading test dataset (partition_id=1)...")
    test_dataset = RAFCEDataset(partition_id=1, use_aligned=True)
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
    
    # Test sample loading
    print("\nTesting sample loading...")
    sample = train_dataset[0]
    print(f"✓ Sample loaded successfully")
    print(f"  - Image ID: {sample['image_id']}")
    print(f"  - Label: {sample['label'].item()}")
    print(f"  - AUs: {sample['aus']}")
    print(f"  - Image shape: {sample['image'].shape}")
    
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    import traceback
    traceback.print_exc()
    all_exist = False

# Test model creation
print("\n" + "=" * 60)
print("Model Creation Test")
print("=" * 60)

try:
    from models.baseline import FERBaseline
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = FERBaseline(model_type='resnet50', num_classes=config['model_config']['num_classes'])
    print(f"✓ Model created successfully")
    print(f"  - Model type: ResNet50")
    print(f"  - Number of classes: {config['model_config']['num_classes']}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
except Exception as e:
    print(f"✗ Error creating model: {e}")
    import traceback
    traceback.print_exc()
    all_exist = False

# Final summary
print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)

if all_exist:
    print("✓ All tests passed! Configuration is working correctly.")
    print("\nYou can now run training with:")
    print("  python kaggle_train.py --model resnet50 --epochs 50 --batch_size 32")
else:
    print("✗ Some tests failed. Please check the errors above.")
    print("\nIf running locally, ensure dataset paths are correct.")
    print("If running on Kaggle, ensure dataset is uploaded with correct structure.")

print("=" * 60)
