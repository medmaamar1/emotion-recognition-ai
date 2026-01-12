# Emotion Recognition AI - Accuracy Improvements

## Overview

This document describes the comprehensive improvements made to the emotion recognition AI system to increase accuracy from below 60% to a target of 75-85%.

## Summary of Improvements

All improvements have been implemented across four main categories:

### 1. Data-Level Improvements ✅
- **Advanced Data Augmentation** with Albumentations
- **Mixup and CutMix** regularization techniques
- **Focal Loss** and **Label Smoothing** for class imbalance

### 2. Model-Level Improvements ✅
- **Advanced Architectures**: ConvNeXt, Swin Transformer, EfficientNet, MobileNetV3
- **Multi-Task Learning**: Emotion + AU prediction with attention mechanisms
- **Ensemble Methods**: Simple, weighted, voting, stacking, and snapshot ensembles

### 3. Training-Level Improvements ✅
- **Advanced LR Schedulers**: Cosine annealing with warmup, OneCycle, polynomial decay
- **Test-Time Augmentation (TTA)** for improved inference
- **Knowledge Distillation** for model compression and transfer learning

### 4. System-Level Improvements ✅
- **Hyperparameter Optimization** with Optuna
- **Cross-Validation** for robust model selection
- **Comprehensive Logging** with TensorBoard and Weights & Biases
- **Per-Class Metrics Analysis** for detailed evaluation
- **Enhanced XAI**: Grad-CAM++, Score-CAM, Integrated Gradients
- **Early Stopping** and **Model Checkpointing**

---

## New Files Created

### Utils Module
- [`utils/__init__.py`](utils/__init__.py) - Package initialization
- [`utils/augmentation.py`](utils/augmentation.py) - Advanced data augmentation with Albumentations
- [`utils/losses.py`](utils/losses.py) - Focal loss, label smoothing, multi-task loss
- [`utils/mixup_cutmix.py`](utils/mixup_cutmix.py) - Mixup and CutMix implementations
- [`utils/logger.py`](utils/logger.py) - Experiment tracking with TensorBoard/W&B

### Models Module
- [`models/multitask.py`](models/multitask.py) - Multi-task learning models
- [`models/advanced_architectures.py`](models/advanced_architectures.py) - ConvNeXt, Swin, EfficientNet
- [`models/ensemble.py`](models/ensemble.py) - Various ensemble methods

### Training Module
- [`training/schedulers.py`](training/schedulers.py) - Advanced LR schedulers
- [`training/tta.py`](training/tta.py) - Test-time augmentation
- [`training/distillation.py`](training/distillation.py) - Knowledge distillation
- [`training/hyperopt.py`](training/hyperopt.py) - Hyperparameter optimization
- [`training/crossval.py`](training/crossval.py) - Cross-validation utilities

### Evaluation Module
- [`evaluation/per_class_analysis.py`](evaluation/per_class_analysis.py) - Per-class metrics analysis
- [`evaluation/xai_enhanced.py`](evaluation/xai_enhanced.py) - Enhanced XAI methods

---

## Usage Examples

### 1. Advanced Data Augmentation

```python
from utils.augmentation import get_train_transform, get_val_transform, get_tta_transforms

# Get advanced augmentation pipeline
train_transform = get_train_transform(img_size=224)
val_transform = get_val_transform(img_size=224)
tta_transforms = get_tta_transforms(img_size=224)

# Apply to dataset
from scripts.dataset import RAFCEDataset
train_dataset = RAFCEDataset(
    DATA_ROOT, PARTITION_FILE, EMOTION_FILE, AU_FILE,
    partition_id=0, transform=train_transform
)
```

### 2. Advanced Loss Functions

```python
from utils.losses import FocalLoss, LabelSmoothingCrossEntropy, MultiTaskLoss

# Focal Loss for class imbalance
criterion = FocalLoss(gamma=2.0, alpha=class_weights)

# Label Smoothing
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

# Multi-Task Loss
criterion = MultiTaskLoss(num_emotions=14, num_aus=18)
total_loss, emotion_loss, au_loss = criterion(emotion_pred, emotion_true, au_pred, au_true)
```

### 3. Mixup and CutMix

```python
from utils.mixup_cutmix import mixup_data, cutmix_data, MixupCutMixScheduler

# Apply Mixup
mixed_x, y_a, y_b, lam = mixup_data(images, labels, alpha=0.2)

# Apply CutMix
mixed_x, y_a, y_b, lam = cutmix_data(images, labels, alpha=1.0)

# Use scheduler for progressive augmentation
scheduler = MixupCutMixScheduler(max_prob=0.5, warmup_epochs=5, total_epochs=100)
```

### 4. Advanced Learning Rate Scheduling

```python
from training.schedulers import CosineAnnealingWarmupScheduler, OneCycleLR

# Cosine Annealing with Warmup
scheduler = CosineAnnealingWarmupScheduler(
    optimizer, warmup_epochs=5, max_epochs=100, min_lr=1e-6
)

# One Cycle Policy
scheduler = OneCycleLR(
    optimizer, max_lr=0.01, total_steps=10000, pct_start=0.3
)
```

### 5. Multi-Task Learning

```python
from models.multitask import FERMultiTask, FERMultiTaskWithAUAttention

# Standard multi-task model
model = FERMultiTask(
    backbone='resnet50',
    num_emotions=14,
    num_aus=18,
    pretrained=True
)

# With AU attention
model = FERMultiTaskWithAUAttention(
    backbone='resnet50',
    num_emotions=14,
    num_aus=18
)

# Forward pass
emotion_logits, au_logits = model(images)
```

### 6. Advanced Model Architectures

```python
from models.advanced_architectures import get_model

# Get ConvNeXt model
model = get_model('convnext_tiny', num_classes=14, pretrained=True)

# Get Swin Transformer
model = get_model('swin_base', num_classes=14, pretrained=True)

# Get EfficientNet
model = get_model('efficientnet_b4', num_classes=14, pretrained=True)
```

### 7. Ensemble Methods

```python
from models.ensemble import FEREnsemble, WeightedEnsemble, StackingEnsemble

# Simple ensemble
ensemble = FEREnsemble([model1, model2, model3], weights=[1.0, 1.0, 1.0])

# Learnable weighted ensemble
ensemble = WeightedEnsemble([model1, model2, model3], num_classes=14)

# Stacking ensemble
ensemble = StackingEnsemble([model1, model2, model3], num_classes=14)

# Predict with ensemble
predictions = ensemble(images)
```

### 8. Test-Time Augmentation

```python
from training.tta import TTAInference, get_tta_transforms

# Get TTA transforms
tta_transforms = get_tta_transforms(img_size=224, mode='standard')

# Create TTA wrapper
tta = TTAInference(model, tta_transforms, device, aggregation_method='mean')

# Predict with TTA
predictions = tta.predict(image)

# With confidence estimation
mean_pred, std_pred = tta.predict_with_confidence(image)
```

### 9. Knowledge Distillation

```python
from training.distillation import distill_model, DistillationLoss

# Distill from teacher to student
student_model = distill_model(
    teacher=teacher_model,
    student=student_model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    epochs=50,
    lr=0.001,
    temperature=4.0,
    alpha=0.7
)

# Or use distillation loss directly
criterion = DistillationLoss(temperature=4.0, alpha=0.7)
total_loss, kd_loss, ce_loss = criterion(student_logits, teacher_logits, labels)
```

### 10. Hyperparameter Optimization

```python
from training.hyperopt import optimize_hyperparameters

# Run hyperparameter optimization
best_params, best_value = optimize_hyperparameters(
    model_class=FERBaseline,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    n_trials=50,
    num_epochs=20
)

print(f"Best parameters: {best_params}")
print(f"Best accuracy: {best_value:.2f}%")
```

### 11. Cross-Validation

```python
from training.crossval import cross_validate

# Perform k-fold cross-validation
results = cross_validate(
    model_class=FERBaseline,
    dataset=train_dataset,
    device=device,
    k=5,
    epochs=20,
    batch_size=32,
    learning_rate=0.001,
    num_classes=14
)

print(f"Mean accuracy: {results['mean_accuracy']:.2f}%")
print(f"Std accuracy: {results['std_accuracy']:.2f}%")
print(f"F1 Macro: {results['f1_macro']:.4f}")
```

### 12. Comprehensive Logging

```python
from utils.logger import ExperimentTracker, EarlyStopping, ModelCheckpoint

# Create experiment tracker
tracker = ExperimentTracker(
    project_name='emotion_recognition',
    experiment_name='experiment_1',
    config={'model': 'resnet50', 'batch_size': 32},
    use_wandb=True
)

# Log metrics
tracker.log_metrics({'train_loss': 0.5, 'train_acc': 75.0}, step=1)
tracker.log_model(model, name='resnet50')
tracker.log_confusion_matrix(y_true, y_pred, class_names, step=10)

# Save checkpoints
tracker.log_checkpoint(model, optimizer, epoch=10, metrics={'val_acc': 80.0})
tracker.log_best_model(model, metrics={'val_acc': 82.5})

# Early stopping
early_stopping = EarlyStopping(patience=10, mode='max')
if early_stopping(val_acc):
    print("Early stopping triggered!")

# Model checkpointing
checkpoint = ModelCheckpoint(save_dir='./checkpoints', monitor='val_acc', mode='max')
checkpoint(model, optimizer, epoch=10, metrics={'val_acc': 82.5})

# Finish experiment
tracker.finish()
```

### 13. Per-Class Analysis

```python
from evaluation.per_class_analysis import PerClassAnalyzer, generate_evaluation_report

# Create analyzer
analyzer = PerClassAnalyzer(num_classes=14)

# Analyze predictions
results = analyzer.analyze(y_true, y_pred, y_prob)

# Get worst performing classes
worst = analyzer.get_worst_performing_classes(y_true, y_pred, metric='f1', top_k=5)

# Get best performing classes
best = analyzer.get_best_performing_classes(y_true, y_pred, metric='f1', top_k=5)

# Generate comprehensive report
generate_evaluation_report(
    y_true, y_pred, y_prob,
    output_dir='./evaluation_results'
)
```

### 14. Enhanced XAI

```python
from evaluation.xai_enhanced import GradCAM, GradCAMPlusPlus, LayerGradCAM, ScoreCAM

# Create Grad-CAM
grad_cam = GradCAM(model, target_layer=model.conv1)
heatmap, pred_class = grad_cam(image)
grad_cam.visualize(heatmap, original_image, save_path='gradcam.png')

# Create Grad-CAM++
grad_cam_plus = GradCAMPlusPlus(model, target_layer=model.conv1)
heatmap_plus, _ = grad_cam_plus(image)

# Layer-wise Grad-CAM
layer_grad_cam = LayerGradCAM(model, [model.conv1, model.conv2])
heatmaps = layer_grad_cam(image)

# Score-CAM
score_cam = ScoreCAM(model, target_layer=model.conv1)
heatmap_score, _ = score_cam(image)

# Generate comprehensive XAI report
from evaluation.xai_enhanced import generate_xai_report
generate_xai_report(model, image, [model.conv1, model.conv2], save_dir='./xai_report')
```

---

## Expected Accuracy Improvements

| Phase | Improvements | Expected Gain | Target Accuracy |
|--------|--------------|----------------|-----------------|
| **Phase 1** | Data augmentation, Mixup/CutMix, Focal loss | +10-15% | 65-70% |
| **Phase 2** | Advanced models, Multi-task, Ensembles | +7-12% | 72-78% |
| **Phase 3** | Better schedulers, TTA, Distillation | +3-7% | 75-82% |
| **Phase 4** | Hyperopt, Cross-val, Logging | +3-5% | 78-85% |

---

## New Dependencies

Add to [`requirements.txt`](requirements.txt):

```txt
# Advanced model architectures
timm>=0.9.0

# Hyperparameter optimization
optuna>=3.0.0

# Experiment tracking
wandb>=0.15.0

# Advanced augmentation
albumentations>=1.3.0

# Additional utilities
pytorch-metric-learning>=2.0.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Training Pipeline Example

Here's a complete training pipeline using all improvements:

```python
import torch
from torch.utils.data import DataLoader
from models.baseline import FERBaseline
from models.multitask import FERMultiTask
from utils.augmentation import get_train_transform, get_val_transform
from utils.losses import FocalLoss, MultiTaskLoss
from utils.mixup_cutmix import MixupCutMixScheduler
from training.schedulers import CosineAnnealingWarmupScheduler
from utils.logger import ExperimentTracker, EarlyStopping, ModelCheckpoint
from scripts.dataset import RAFCEDataset

# Configuration
config = {
    'model': 'resnet50',
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'use_mixup': True,
    'use_focal_loss': True,
    'use_tta': True
}

# Create experiment tracker
tracker = ExperimentTracker(
    project_name='emotion_recognition',
    experiment_name='improved_model',
    config=config,
    use_wandb=True
)

# Load data
train_transform = get_train_transform(img_size=224)
val_transform = get_val_transform(img_size=224)

train_dataset = RAFCEDataset(DATA_ROOT, PARTITION_FILE, EMOTION_FILE, AU_FILE,
                           partition_id=0, transform=train_transform)
val_dataset = RAFCEDataset(DATA_ROOT, PARTITION_FILE, EMOTION_FILE, AU_FILE,
                         partition_id=2, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Create model
model = FERBaseline(model_type='resnet50', num_classes=14).to(device)

# Loss function
criterion = FocalLoss(gamma=2.0)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Scheduler
scheduler = CosineAnnealingWarmupScheduler(optimizer, warmup_epochs=5, max_epochs=100)

# Mixup scheduler
mixup_scheduler = MixupCutMixScheduler(max_prob=0.5, warmup_epochs=5, total_epochs=100)

# Early stopping
early_stopping = EarlyStopping(patience=15, mode='max')

# Model checkpointing
checkpoint = ModelCheckpoint(save_dir='./checkpoints', monitor='val_acc', mode='max')

# Training loop
best_val_acc = 0.0

for epoch in range(100):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch in train_loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Apply Mixup
        if mixup_scheduler.should_apply():
            images, y_a, y_b, lam = mixup_data(images, labels)
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_acc = 100 * train_correct / train_total
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_acc = 100 * val_correct / val_total
    
    # Log metrics
    tracker.log_metrics({
        'train_loss': train_loss / len(train_loader),
        'train_acc': train_acc,
        'val_loss': val_loss / len(val_loader),
        'val_acc': val_acc
    }, step=epoch)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        tracker.log_best_model(model, metrics={'val_acc': val_acc})
    
    # Checkpoint
    checkpoint(model, optimizer, epoch, metrics={'val_acc': val_acc})
    
    # Early stopping
    if early_stopping(val_acc):
        print("Early stopping triggered!")
        break
    
    # Update schedulers
    scheduler.step()
    mixup_scheduler.step()

# Finish experiment
tracker.finish()
print(f"Best validation accuracy: {best_val_acc:.2f}%")
```

---

## Next Steps

1. **Install new dependencies**: `pip install -r requirements.txt`
2. **Run baseline experiments**: Establish new baseline with improvements
3. **Hyperparameter optimization**: Find optimal hyperparameters
4. **Train multiple models**: Compare different architectures
5. **Create ensemble**: Combine best models
6. **Evaluate thoroughly**: Use per-class analysis and XAI
7. **Deploy best model**: Use TTA for production inference

---

## File Structure

```
emotion recognition ai/
├── utils/
│   ├── __init__.py (new)
│   ├── augmentation.py (new)
│   ├── losses.py (new)
│   ├── mixup_cutmix.py (new)
│   └── logger.py (new)
├── models/
│   ├── baseline.py (existing)
│   ├── vision_llm.py (existing)
│   ├── multitask.py (new)
│   ├── advanced_architectures.py (new)
│   └── ensemble.py (new)
├── training/
│   ├── train_baseline.py (existing - enhance)
│   ├── train_vllm.py (existing - enhance)
│   ├── schedulers.py (new)
│   ├── tta.py (new)
│   ├── distillation.py (new)
│   ├── hyperopt.py (new)
│   └── crossval.py (new)
├── evaluation/
│   ├── metrics.py (existing)
│   ├── xai_gradcam.py (existing)
│   ├── per_class_analysis.py (new)
│   └── xai_enhanced.py (new)
├── scripts/
│   ├── dataset.py (existing)
│   └── analyze_distribution.py (existing)
├── app/
│   └── main.py (existing)
├── plans/
│   └── accuracy_improvement_plan.md (new)
├── requirements.txt (update)
├── README.md (existing)
└── IMPROVEMENTS.md (this file)
```

---

## Performance Tracking

Track your progress with the following metrics:

- **Training Metrics**: Loss, accuracy, learning rate, gradient norms
- **Validation Metrics**: Loss, accuracy, F1 score (macro/weighted)
- **Per-Class Metrics**: Precision, recall, F1 for each of 14 emotions
- **Inference Metrics**: Time per image, memory usage, TTA speedup
- **Model Metrics**: Parameter count, FLOPs, model size

---

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use gradient accumulation
- Enable gradient checkpointing
- Use mixed precision training (FP16)

### Slow Training
- Use `num_workers > 0` in DataLoader
- Enable pin_memory=True
- Use mixed precision training
- Reduce TTA transforms

### Overfitting
- Increase dropout
- Add more augmentation
- Use stronger regularization (weight decay)
- Implement early stopping

### Underfitting
- Increase model capacity
- Train for more epochs
- Reduce regularization
- Check data quality

---

## References

- [Albumentations Documentation](https://albumentations.ai/docs/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [Mixup Paper](https://arxiv.org/abs/1710.09412)
- [CutMix Paper](https://arxiv.org/abs/1905.04899)
- [Focal Loss Paper](https://arxiv.org/abs/1708.02002)

---

## License

This project follows the same license as the original emotion recognition AI project.

## Contact

For questions or issues, please refer to the project repository.
