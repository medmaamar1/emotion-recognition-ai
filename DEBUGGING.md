# Debugging Guide for Emotion Recognition AI

This guide helps you identify and fix potential bugs when running the emotion recognition code.

## Common Error Messages and Solutions

### 1. Import Errors

#### Error: `ModuleNotFoundError: No module named 'timm'`
```
Solution: Install timm package
pip install timm>=0.9.0
```

#### Error: `ModuleNotFoundError: No module named 'optuna'`
```
Solution: Install optuna package
pip install optuna>=3.0.0
```

#### Error: `ModuleNotFoundError: No module named 'albumentations'`
```
Solution: Install albumentations package
pip install albumentations>=1.3.0
```

#### Error: `ModuleNotFoundError: No module named 'wandb'`
```
Solution: Install wandb package (optional)
pip install wandb>=0.15.0
```

### 2. CUDA/GPU Errors

#### Error: `RuntimeError: CUDA out of memory`
```
Cause: GPU memory exhausted

Solutions:
1. Reduce batch size in training script
2. Use smaller model (e.g., convnext_tiny instead of convnext_base)
3. Reduce image size (e.g., 224x224 instead of 384x384)
4. Enable gradient accumulation
5. Clear cache: torch.cuda.empty_cache()
```

#### Error: `RuntimeError: Expected all tensors to be on the same device`
```
Cause: Model and data on different devices (CPU vs GPU)

Solution: Ensure both model and data are on same device
model = model.to(device)
images = images.to(device)
labels = labels.to(device)
```

#### Error: `AssertionError: Torch not compiled with CUDA enabled`
```
Cause: PyTorch CPU-only version installed

Solution: Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Data Loading Errors

#### Error: `FileNotFoundError: [Errno 2] No such file or directory`
```
Cause: Dataset path incorrect

Solution: Check dataset path in training script
1. Verify RAF-CE dataset is in correct location
2. Update data_root parameter to point to dataset
3. Ensure image files exist and are accessible
```

#### Error: `ValueError: Image has wrong dimensions`
```
Cause: Image size mismatch with expected input

Solution: Ensure images are properly resized
1. Check augmentation pipeline image_size parameter
2. Verify all images are RGB (3 channels)
3. Check for corrupted images in dataset
```

#### Error: `RuntimeError: DataLoader worker exited unexpectedly`
```
Cause: Data loading worker crashed

Solution: Reduce num_workers or set to 0
train_loader = DataLoader(dataset, batch_size=32, num_workers=0)
```

### 4. Model Architecture Errors

#### Error: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`
```
Cause: Input size doesn't match model expected size

Solution: Check model input dimensions
1. Verify image size matches model expectation (e.g., 224x224)
2. Check model's num_classes parameter (should be 14 for RAF-CE)
3. Ensure batch dimension is included: (N, C, H, W)
```

#### Error: `KeyError: 'Unexpected key(s) in state_dict'`
```
Cause: Loading checkpoint from different model architecture

Solution: Load checkpoint with strict=False
model.load_state_dict(torch.load(checkpoint_path), strict=False)
```

#### Error: `AttributeError: 'ViTModel' object has no attribute 'fc'`
```
Cause: Incorrectly accessing ViT attributes

Solution: Use correct ViT API
# For transformers ViT:
features = model.vit(x).last_hidden_state[:, 0]
# For timm ViT:
features = model(x)
```

### 5. Training Errors

#### Error: `RuntimeError: element 0 of tensors does not require grad`
```
Cause: Trying to optimize frozen parameters

Solution: Ensure parameters require gradients
for param in model.parameters():
    param.requires_grad = True
```

#### Error: `ValueError: Expected input batch_size (X) to match target batch_size (Y)`
```
Cause: Batch size mismatch between images and labels

Solution: Check data loading pipeline
1. Ensure dataset returns correct format
2. Verify batch size is consistent
3. Check for None values in batch
```

#### Error: `RuntimeError: Trying to backward through the graph a second time`
```
Cause: Calling backward() multiple times on same computation graph

Solution: Clear gradients before backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 6. Loss Function Errors

#### Error: `RuntimeError: CUDA error: device-side assert triggered`
```
Cause: Invalid target labels (out of range)

Solution: Verify label values are in valid range [0, num_classes-1]
assert torch.all(labels >= 0) and torch.all(labels < num_classes)
```

#### Error: `ValueError: Target and input must have the same number of elements`
```
Cause: Label shape doesn't match predictions

Solution: Ensure correct label format
# For single-label classification:
labels = torch.randint(0, num_classes, (batch_size,))
# For multi-label classification:
labels = torch.randint(0, 2, (batch_size, num_classes))
```

### 7. Scheduler Errors

#### Error: `ValueError: warmup_epochs must be less than max_epochs`
```
Cause: Invalid scheduler parameters

Solution: Adjust warmup and max_epochs
scheduler = CosineAnnealingWarmupScheduler(
    optimizer,
    warmup_epochs=5,  # Must be < max_epochs
    max_epochs=100
)
```

#### Error: `UserWarning: Detected call of lr_scheduler.step() before optimizer.step()`
```
Cause: Calling scheduler before optimizer in first iteration

Solution: Call optimizer.step() before scheduler.step()
optimizer.step()
scheduler.step()
```

## Debugging Tips

### 1. Enable Verbose Logging

Add these lines at the start of your script:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
torch.set_printoptions(profile='full')
```

### 2. Use TensorBoard for Monitoring

```python
from utils.logger import ExperimentTracker

tracker = ExperimentTracker(
    log_dir='./logs',
    use_tensorboard=True,
    use_wandb=False
)

# Log metrics
tracker.log_scalar('train/loss', loss, epoch)
tracker.log_scalar('train/accuracy', accuracy, epoch)
```

### 3. Add Debug Prints

```python
# Print tensor shapes
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Output shape: {outputs.shape}")

# Print device info
print(f"Model device: {next(model.parameters()).device}")
print(f"Images device: {images.device}")

# Print loss components
loss, kd_loss, ce_loss = criterion(student_output, teacher_output, labels)
print(f"Total loss: {loss.item():.4f}, KD: {kd_loss.item():.4f}, CE: {ce_loss.item():.4f}")
```

### 4. Validate Dataset

```python
from scripts.dataset import RAFCEDataset

# Create dataset
dataset = RAFCEDataset(
    data_root='./RAF-CE',
    split='train',
    transform=None
)

# Check first sample
sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")
print(f"Label: {sample['label']}")
print(f"Label value range: {dataset.labels.min()}-{dataset.labels.max()}")

# Check for corrupted images
for i in range(len(dataset)):
    try:
        sample = dataset[i]
    except Exception as e:
        print(f"Corrupted sample at index {i}: {e}")
```

### 5. Test Model Forward Pass

```python
# Create dummy input
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 4
dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)

# Test model
model = FERConvNeXt('convnext_tiny', num_classes=14).to(device)
model.eval()

with torch.no_grad():
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output range: {output.min():.4f} - {output.max():.4f}")
    print(f"Output device: {output.device}")
```

### 6. Check Memory Usage

```python
import torch

def print_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    else:
        print("CUDA not available")

print_memory_usage()
```

## Testing Checklist

Before running full training, run these tests:

### 1. Import Test
```python
# Test all imports
import torch
from utils.augmentation import get_train_transform, get_val_transform
from utils.losses import FocalLoss, LabelSmoothingCrossEntropy
from models.multitask import FERMultiTask
from models.advanced_architectures import FERConvNeXt, FERSwin
from models.ensemble import FEREnsemble
from training.schedulers import CosineAnnealingWarmupScheduler
from training.tta import TTAInference
from training.distillation import DistillationLoss
from training.hyperopt import HyperparameterOptimizer
from evaluation.per_class_analysis import PerClassAnalyzer
from evaluation.xai_enhanced import GradCAM
print("All imports successful!")
```

### 2. Dataset Test
```python
from scripts.dataset import RAFCEDataset
from torch.utils.data import DataLoader

# Test dataset loading
dataset = RAFCEDataset(
    data_root='./RAF-CE',
    split='train',
    transform=get_train_transform(224)
)
print(f"Dataset size: {len(dataset)}")

# Test dataloader
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
batch = next(iter(loader))
print(f"Batch images shape: {batch['image'].shape}")
print(f"Batch labels shape: {batch['label'].shape}")
```

### 3. Model Test
```python
# Test model forward pass
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FERConvNeXt('convnext_tiny', num_classes=14).to(device)
dummy_input = torch.randn(2, 3, 224, 224).to(device)
output = model(dummy_input)
print(f"Model output shape: {output.shape}")
assert output.shape == (2, 14), "Output shape mismatch!"
```

### 4. Training Step Test
```python
# Test single training step
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

dummy_input = torch.randn(4, 3, 224, 224).to(device)
dummy_labels = torch.randint(0, 14, (4,)).to(device)

optimizer.zero_grad()
output = model(dummy_input)
loss = criterion(output, dummy_labels)
loss.backward()
optimizer.step()

print(f"Training step successful! Loss: {loss.item():.4f}")
```

## Common Issues and Solutions

### Issue: Training loss not decreasing
**Possible causes:**
1. Learning rate too high or too low
2. Model architecture too complex/small for dataset
3. Data augmentation too aggressive
4. Incorrect loss function

**Solutions:**
1. Try learning rate 1e-4 to 1e-3
2. Use smaller model (convnext_tiny)
3. Reduce augmentation intensity
4. Use FocalLoss for class imbalance

### Issue: Overfitting (train loss decreases, val loss increases)
**Possible causes:**
1. Model too complex
2. Not enough training data
3. Insufficient regularization

**Solutions:**
1. Add dropout or increase dropout rate
2. Use label smoothing
3. Add Mixup/CutMix augmentation
4. Use early stopping

### Issue: Underfitting (both train and val loss high)
**Possible causes:**
1. Model too simple
2. Learning rate too low
3. Not enough training epochs

**Solutions:**
1. Use larger model (convnext_base, swin_base)
2. Increase learning rate
3. Train for more epochs
4. Use cosine annealing with warmup

### Issue: Large variance between runs
**Possible causes:**
1. Random seed not set
2. Data shuffling
3. Dropout randomness

**Solutions:**
```python
# Set random seeds for reproducibility
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

## Getting Help

If you encounter an error not listed here:

1. **Copy the full error traceback** - Include all lines from the error
2. **Check the error type** - RuntimeError, ValueError, etc.
3. **Identify the line number** - See which line caused the error
4. **Check variable values** - Print shapes, types, and values of variables
5. **Search the error message** - Many errors have known solutions

## Performance Profiling

To identify bottlenecks in your code:

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True
) as prof:
    # Your training code here
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```
