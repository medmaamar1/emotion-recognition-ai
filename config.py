"""
Configuration file for Emotion Recognition AI Project.
Supports both local and Kaggle environments.
"""

import os
from pathlib import Path

# Detect if running on Kaggle
IS_KAGGLE = os.path.exists('/kaggle/input')

if IS_KAGGLE:
    # Kaggle paths
    BASE_DIR = '/kaggle/input'
    DATA_DIR = '/kaggle/input'
    
    # Dataset paths on Kaggle
    DATA_ROOT_RAW = '/kaggle/input/emotion-images/RAF-AU/original/original'
    DATA_ROOT_ALIGNED = '/kaggle/input/emotion-images/RAF-AU/aligned/aligned'
    
    # Label and metadata files
    PARTITION_FILE = '/kaggle/input/emotions-augmentations/RAFCE_partition.txt'
    EMOTION_FILE = '/kaggle/input/emotions-augmentations/RAFCE_emolabel.txt'
    AU_FILE = '/kaggle/input/emotions-augmentations/RAFCE_AUlabel.txt'
    
    # Additional files
    DISTRIBUTION_FILE = '/kaggle/input/distibution/distribution.txt'
    REQUIREMENTS_FILE = '/kaggle/input/requirement/requirements.txt'
    
    # Output directory for models and results
    OUTPUT_DIR = '/kaggle/working'
    CHECKPOINT_DIR = '/kaggle/working/checkpoints'
    LOG_DIR = '/kaggle/working/logs'
    
else:
    # Local paths (Windows)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'dataset_root')
    
    # Dataset paths locally
    DATA_ROOT_RAW = os.path.join(BASE_DIR, 'dataset_root', 'RAF-AU', 'original')
    DATA_ROOT_ALIGNED = os.path.join(BASE_DIR, 'RAF-AU', 'aligned')
    
    # Label and metadata files
    PARTITION_FILE = os.path.join(BASE_DIR, 'RAFCE_partition.txt')
    EMOTION_FILE = os.path.join(BASE_DIR, 'RAFCE_emolabel.txt')
    AU_FILE = os.path.join(BASE_DIR, 'RAFCE_AUlabel.txt')
    
    # Additional files
    DISTRIBUTION_FILE = os.path.join(BASE_DIR, 'distribution.txt')
    REQUIREMENTS_FILE = os.path.join(BASE_DIR, 'requirements.txt')
    
    # Output directory for models and results
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')

# Create necessary directories if they don't exist
for directory in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'num_workers': 4 if not IS_KAGGLE else 2,  # Reduce workers on Kaggle
    'pin_memory': True,
    'device': 'cuda' if not IS_KAGGLE else 'cuda',  # Will be set based on availability
}

# Model configuration
MODEL_CONFIG = {
    'num_classes': 14,  # 14 compound emotion categories
    'num_au': 18,  # Number of Action Units
    'backbone': 'resnet50',  # Options: resnet50, efficientnet_b0, vit_base_patch16_224
    'pretrained': True,
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    'use_mixup': True,
    'use_cutmix': True,
    'mixup_alpha': 0.2,
    'cutmix_alpha': 1.0,
    'cutmix_prob': 0.5,
}

# Partition IDs
PARTITION = {
    'train': 0,
    'test': 1,
    'validation': 2,
}

# Emotion labels mapping
EMOTION_LABELS = {
    0: 'Happily surprised',
    1: 'Happily disgusted',
    2: 'Sadly fearful',
    3: 'Sadly angry',
    4: 'Sadly surprised',
    5: 'Sadly disgusted',
    6: 'Fearfully angry',
    7: 'Fearfully surprised',
    8: 'Fearfully disgusted',
    9: 'Angrily surprised',
    10: 'Angrily disgusted',
    11: 'Disgustedly surprised',
    12: 'Happily fearful',
    13: 'Happily sad',
}

# Action Units (AU) list
AU_LABELS = [
    'AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10',
    'AU12', 'AU15', 'AU17', 'AU20', 'AU23', 'AU24', 'AU25', 'AU26', 'AU43', 'AU45'
]

def get_config():
    """Return the configuration dictionary."""
    return {
        'is_kaggle': IS_KAGGLE,
        'data_root_raw': DATA_ROOT_RAW,
        'data_root_aligned': DATA_ROOT_ALIGNED,
        'partition_file': PARTITION_FILE,
        'emotion_file': EMOTION_FILE,
        'au_file': AU_FILE,
        'output_dir': OUTPUT_DIR,
        'checkpoint_dir': CHECKPOINT_DIR,
        'log_dir': LOG_DIR,
        'train_config': TRAIN_CONFIG,
        'model_config': MODEL_CONFIG,
        'augmentation_config': AUGMENTATION_CONFIG,
        'partition': PARTITION,
        'emotion_labels': EMOTION_LABELS,
        'au_labels': AU_LABELS,
    }

if __name__ == "__main__":
    # Print configuration for verification
    config = get_config()
    print("=" * 60)
    print("Configuration Settings")
    print("=" * 60)
    print(f"Running on Kaggle: {config['is_kaggle']}")
    print(f"Data Root (Raw): {config['data_root_raw']}")
    print(f"Data Root (Aligned): {config['data_root_aligned']}")
    print(f"Partition File: {config['partition_file']}")
    print(f"Emotion File: {config['emotion_file']}")
    print(f"AU File: {config['au_file']}")
    print(f"Output Directory: {config['output_dir']}")
    print(f"Checkpoint Directory: {config['checkpoint_dir']}")
    print(f"Log Directory: {config['log_dir']}")
    print("=" * 60)
