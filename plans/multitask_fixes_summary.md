# Multi-Task Training Fixes Summary

## Problem Analysis

### Issue 1: AU Prediction Failure (AU-F1 = 0.0000)
**Root Cause:**
- AU labels in [`RAFCE_AUlabel.txt`](../RAFCE_AUlabel.txt) contain many "null" entries
- Model learned to predict "no AUs" for everything to minimize loss
- AU loss (0.0002-0.0004) was 1000x smaller than emotion loss (2.44-2.90)
- AU loss weight (0.5) was too low to provide meaningful supervision

**Evidence from Training Logs:**
```
Epoch 50/50: Train Loss: 2.5436, Train Acc: 96.60%, 
Val Loss: 2.8943, Val Acc: 51.99%, 
Macro-F1: 0.4255, AU-F1: 0.0000
```

### Issue 2: Severe Overfitting (Train-Val Gap: 44.62%)
**Root Cause:**
- Training: 96.60%, Validation: 51.99%
- Train-val gap of 44.62% indicates severe overfitting
- Model memorizes training data but fails to generalize
- Regularization was too weak for small dataset (~2,700 samples)

**Evidence from Training Logs:**
```
Train Acc: 96.60%
Val Acc: 51.99%
Train-Val Gap: 44.62%
Minority Class F1: [4:0.00, 8:0.00, 12:0.00, 13:0.00]
```

## Fixes Implemented

### Fix 1: Handle Null AU Labels Properly
**Problem:** AU loss was computed for all samples, including those with "null" labels
**Solution:** Only compute AU loss for samples with valid AU labels

**Implementation:**
```python
# Create mask for valid AU labels
au_valid_mask = (au_labels_str != "null").float()

if au_valid_mask.sum() > 0:
    # Only compute AU loss for valid samples
    au_logits_valid = au_logits[au_valid_mask == 1]
    au_labels_valid = au_labels[au_valid_mask == 1]
    au_loss_valid = au_criterion(au_logits_valid, au_labels_valid)
    
    # Scale AU loss to account for valid samples
    au_loss = au_loss_valid * (au_valid_mask.sum() / au_valid_mask.numel())
else:
    # No valid AUs, no AU loss
    au_loss = torch.tensor(0.0, device=device)
```

**Expected Impact:** AU-F1 should increase from 0.0000 to >0.50

---

### Fix 2: Balance AU Loss Weight
**Problem:** AU loss weight (0.5) was too low
**Solution:** Increase to 1.0-2.0 for meaningful AU prediction

**Implementation:**
```python
parser.add_argument('--au_loss_weight', type=float, default=1.0,
                   help='Weight for AU loss (default: 1.0, increased for meaningful AU prediction)')
```

**Expected Impact:** AU loss becomes comparable to emotion loss, enabling meaningful AU supervision

---

### Fix 3: Stronger Regularization to Reduce Overfitting
**Problem:** Regularization was too weak for small dataset
**Solution:** Increase all regularization parameters

**Implementation:**

#### 3.1 Increased Dropout Rate
```python
parser.add_argument('--dropout_rate', type=float, default=0.5,
                   help='Dropout rate (default: 0.5, increased to reduce overfitting)')
```
**Change:** 0.3 → 0.5 (67% increase)

#### 3.2 Increased Weight Decay
```python
parser.add_argument('--weight_decay', type=float, default=1e-3,
                   help='Weight decay (default: 1e-3, increased for regularization)')
```
**Change:** 1e-4 → 1e-3 (10x increase)

#### 3.3 Added Gradient Clipping
```python
parser.add_argument('--grad_clip', type=float, default=1.0,
                   help='Gradient clipping max norm (default: 1.0)')
```
**Change:** None → 1.0 (new feature)

#### 3.4 Reduced Learning Rate
```python
parser.add_argument('--lr', type=float, default=0.0005, 
                   help='Learning rate (default: 0.0005, reduced for stability)')
```
**Change:** 0.001 → 0.0005 (50% reduction)

#### 3.5 Added Label Smoothing
```python
parser.add_argument('--label_smoothing', type=float, default=0.1,
                   help='Label smoothing factor (default: 0.1, reduces overfitting)')
```
**Change:** None → 0.1 (new feature)

#### 3.6 Increased Early Stopping Patience
```python
parser.add_argument('--patience', type=int, default=20,
                   help='Early stopping patience (default: 20, increased)')
```
**Change:** 15 → 20 (33% increase)

**Expected Impact:** Train-val gap should reduce from 44.62% to <25%

---

### Fix 4: Better AU Label Parsing
**Problem:** AU label parsing didn't handle all format variations
**Solution:** Already implemented in [`scripts/dataset.py`](../scripts/dataset.py:17)

**Implementation:**
```python
def parse_au_labels(au_str):
    """
    Parse AU string labels to binary vector.
    
    Args:
        au_str: String like "AU1 AU2 AU4" or "null"
    
    Returns:
        Binary vector of length 18 indicating which AUs are present
    """
    au_vector = np.zeros(len(AU_LABELS), dtype=np.float32)
    
    if au_str == "null" or au_str is None:
        return au_vector
    
    aus = au_str.split()
    for au in aus:
        if au in AU_LABELS:
            idx = AU_LABELS.index(au)
            au_vector[idx] = 1.0
    
    return au_vector
```

**AU Labels List:**
```python
AU_LABELS = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10',
              'AU12', 'AU15', 'AU17', 'AU20', 'AU23', 'AU24', 'AU25', 'AU26', 'AU43', 'AU45']
```

**Expected Impact:** All AU label formats are correctly parsed

---

## Expected Results

### Before Fixes (kaggle_train_multitask.py)
```
Validation Accuracy: 51.99%
Macro-F1: 0.4255
AU-F1: 0.0000
Train-Val Gap: 44.62%
Minority Class F1: [4:0.00, 8:0.00, 12:0.00, 13:0.00]
```

### After Fixes (kaggle_train_multitask_fixed.py)
**Expected:**
```
Validation Accuracy: 55-60%
Macro-F1: 0.50-0.60
AU-F1: 0.50-0.70
Train-Val Gap: 20-25%
Minority Class F1: [4:0.20-0.40, 8:0.15-0.35, 12:0.20-0.40, 13:0.20-0.40]
```

---

## How to Run Fixed Training

### Basic Command (Recommended)
```bash
python kaggle_train_multitask_fixed.py \
    --model resnet50 \
    --epochs 50 \
    --batch_size 32 \
    --au_loss_weight 1.0 \
    --dropout_rate 0.5 \
    --weight_decay 1e-3 \
    --grad_clip 1.0 \
    --label_smoothing 0.1 \
    --patience 20 \
    --monitor_metric macro_f1
```

### With AU Attention (Advanced)
```bash
python kaggle_train_multitask_fixed.py \
    --model resnet50 \
    --epochs 50 \
    --batch_size 32 \
    --au_loss_weight 1.0 \
    --use_au_attention \
    --dropout_rate 0.5 \
    --weight_decay 1e-3 \
    --grad_clip 1.0 \
    --label_smoothing 0.1 \
    --patience 20 \
    --monitor_metric macro_f1
```

### Minimal Augmentation (If Overfitting Persists)
```bash
python kaggle_train_multitask_fixed.py \
    --model resnet50 \
    --epochs 50 \
    --batch_size 32 \
    --au_loss_weight 1.0 \
    --no_augmentation \
    --dropout_rate 0.6 \
    --weight_decay 2e-3 \
    --grad_clip 0.5 \
    --label_smoothing 0.2 \
    --patience 25 \
    --monitor_metric macro_f1
```

---

## Key Changes Summary

| Parameter | Before | After | Change | Impact |
|-----------|---------|--------|---------|
| AU Loss Weight | 0.5 | 1.0 | +100% | AU-F1: 0.0000 → >0.50 |
| Dropout Rate | 0.3 | 0.5 | +67% | Reduces overfitting |
| Weight Decay | 1e-4 | 1e-3 | +900% | Reduces overfitting |
| Learning Rate | 0.001 | 0.0005 | -50% | Improves stability |
| Gradient Clipping | None | 1.0 | New | Prevents exploding gradients |
| Label Smoothing | None | 0.1 | New | Reduces overfitting |
| Early Stopping Patience | 15 | 20 | +33% | More training time |
| Null AU Handling | No | Yes | New | AU-F1: 0.0000 → >0.50 |

---

## Monitoring During Training

### Key Metrics to Watch

1. **AU-F1 Score**
   - Should increase from 0.0000 to >0.50
   - If still 0.0000 after 5 epochs, increase `--au_loss_weight` to 2.0

2. **Train-Val Gap**
   - Should decrease from 44.62% to <25%
   - If still >30%, increase `--dropout_rate` to 0.6 or `--weight_decay` to 2e-3

3. **Minority Class F1**
   - Should increase from 0.00 to >0.20
   - If still 0.00, consider oversampling minority classes

4. **Emotion Loss vs AU Loss**
   - Should be balanced (within 10x of each other)
   - If AU loss is still 1000x smaller, increase `--au_loss_weight`

---

## Troubleshooting

### If AU-F1 Remains 0.0000
**Possible Causes:**
1. AU loss weight still too low
2. Too many null AU labels
3. AU labels are noisy

**Solutions:**
1. Increase `--au_loss_weight` to 2.0 or 3.0
2. Check AU label distribution in logs
3. Consider removing samples with null AU labels from training

### If Train-Val Gap Remains >30%
**Possible Causes:**
1. Regularization still too weak
2. Model capacity too high
3. Dataset too small

**Solutions:**
1. Increase `--dropout_rate` to 0.6 or 0.7
2. Increase `--weight_decay` to 2e-3 or 5e-3
3. Reduce `--grad_clip` to 0.5
4. Try smaller model (e.g., resnet18 instead of resnet50)

### If Validation Accuracy Decreases
**Possible Causes:**
1. Learning rate too low
2. Regularization too strong
3. AU loss weight too high

**Solutions:**
1. Increase `--lr` to 0.001
2. Decrease `--dropout_rate` to 0.4
3. Decrease `--weight_decay` to 5e-4
4. Decrease `--au_loss_weight` to 0.5

---

## Next Steps After Successful Training

1. **Compare with Baseline (55%)**
   - If fixed multi-task >55%, AU supervision is working
   - If still <55%, consider alternative approaches

2. **Try AU Attention Model**
   - Run with `--use_au_attention` flag
   - AU-guided attention may improve performance

3. **Hyperparameter Tuning**
   - Experiment with different AU loss weights (0.5, 1.0, 2.0)
   - Try different dropout rates (0.4, 0.5, 0.6)
   - Try different weight decay values (5e-4, 1e-3, 2e-3)

4. **If Performance Still Poor**
   - Consider using AU labels as weak supervision only
   - Try curriculum learning (start with emotion, add AU later)
   - Consider alternative approaches (face pretraining, attention mechanisms)

---

## Files Modified

1. **kaggle_train_multitask_fixed.py** - Fixed multi-task training script
   - Handles null AU labels
   - Balanced AU loss weight
   - Stronger regularization
   - Better logging

2. **scripts/dataset.py** - Already has correct AU label parsing
   - Returns both 'aus' (original string) and 'au_vector' (binary)
   - Handles all AU label formats

---

## References

- Original multi-task script: [`kaggle_train_multitask.py`](../kaggle_train_multitask.py)
- Fixed multi-task script: [`kaggle_train_multitask_fixed.py`](../kaggle_train_multitask_fixed.py)
- Dataset with AU labels: [`scripts/dataset.py`](../scripts/dataset.py)
- AU labels file: [`RAFCE_AUlabel.txt`](../RAFCE_AUlabel.txt)
- Multi-task models: [`models/multitask.py`](../models/multitask.py)
