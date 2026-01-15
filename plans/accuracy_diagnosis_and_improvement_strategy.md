# Accuracy Diagnosis and Improvement Strategy

## Current Situation

**Validation Accuracy: 55%** (up from 48.87%, but still very low)

## Critical Issues Identified

### 1. **NOT Using AU Labels** (Major Issue)
The dataset provides **Action Unit (AU) labels** for each image, but the current training completely ignores them.

**What are AUs?**
- Action Units are facial muscle movements (e.g., AU1 = Inner Brow Raiser, AU12 = Lip Corner Puller)
- 18 AUs are provided per image
- AUs are the **ground truth of facial expressions** - they directly indicate which muscles are activated
- Emotions are COMPOSED of specific AU combinations

**Why This Matters:**
- AUs provide **explicit supervision** about facial muscle movements
- They are **fine-grained annotations** that can guide the model to focus on relevant facial regions
- Multi-task learning with AUs has been shown to **significantly improve emotion recognition** (research papers report 10-20% gains)
- AUs help the model learn **what to look for** in the face

**Current State:**
- Dataset loads AU labels (see [`scripts/dataset.py`](scripts/dataset.py:93)) but they're returned as strings and never used
- Model only takes images as input, ignores AU supervision
- Multi-task model exists ([`models/multitask.py`](models/multitask.py:1)) but is not used in training

### 2. **Suboptimal Model Architecture**
**Current:** ResNet50 pretrained on ImageNet
- ImageNet features are for object/scene recognition, not facial expressions
- ResNet50 doesn't have attention mechanisms to focus on facial regions
- No explicit modeling of facial structure

**Better Approaches:**
- **Face-specific backbones**: Models pretrained on face datasets (e.g., FaceNet, ArcFace)
- **Attention mechanisms**: Spatial attention to focus on eyes, mouth, nose regions
- **Vision Transformers**: Better at capturing global relationships
- **Multi-scale features**: Combine local (eye region) and global (whole face) features

### 3. **No Feature Engineering**
**Current:** Raw pixels → ResNet50 → classifier

**Missing Features:**
- **AU-based features**: Convert AU string labels to binary vectors and use as additional input
- **Facial landmark features**: 68-point landmarks could guide attention
- **Optical flow**: Temporal information (if video data available)
- **Histogram features**: Color histograms, edge histograms
- **Texture features**: LBP, Gabor filters for facial textures

### 4. **Generic Data Augmentation**
**Current:** Random flip, rotation, color jitter - all generic image augmentations

**Problem:** These don't preserve facial expression semantics
- Random rotation might distort facial geometry
- Color jitter might affect skin tone recognition

**Better Approaches:**
- **Face-aware augmentation**: Only augment regions that don't affect expression
- **Expression-preserving transforms**: Small rotations, brightness changes
- **GAN-based augmentation**: Generate synthetic samples for minority classes
- **CutMix/MixUp**: Already implemented but need careful tuning

### 5. **Severe Class Imbalance**
**Current Distribution:**
- Class 10: 582 samples (21.48%)
- Class 5: 495 samples (18.27%)
- Class 12: 6 samples (0.22%) - EXTREMELY LOW
- Class 13: 17 samples (0.63%) - VERY LOW

**Current Fixes (Not Working Well):**
- WeightedRandomSampler - causes over-sampling of rare classes
- Focal Loss - too aggressive, caused 8.16% accuracy

**Better Approaches:**
- **Curriculum learning**: Start with majority classes, gradually introduce minority classes
- **Few-shot learning**: Treat minority classes as few-shot problems
- **Meta-learning**: Learn to learn from few examples
- **Data augmentation**: Oversample minority classes with smart augmentation
- **Class-balanced loss**: Better than focal loss for extreme imbalance

### 6. **Single-Task Learning**
**Current:** Only predict emotion class

**Better:** Multi-task learning
- Predict emotion + AUs simultaneously
- AU prediction provides auxiliary supervision
- Shared features benefit both tasks
- Research shows 5-15% improvement

## Proposed Improvement Strategy

### Phase 1: Quick Wins (Use Existing Assets)

#### 1.1 Parse and Use AU Labels
- Convert AU string labels to binary vectors (18-dimensional)
- Parse AU labels in dataset: "AU1 AU2 AU4" → [1,1,1,0,...,0]
- Use AUs as additional supervision signal

#### 1.2 Implement Multi-Task Learning
- Use existing [`FERMultiTask`](models/multitask.py:11) model
- Train to predict both emotions AND AUs
- Loss = emotion_loss + λ * au_loss
- Expected gain: 5-10% accuracy

#### 1.3 Add AU-Guided Attention
- Use existing [`FERMultiTaskWithAUAttention`](models/multitask.py:142) model
- Attention mechanism focuses on AU-relevant regions
- Expected gain: 3-5% accuracy

### Phase 2: Architecture Improvements

#### 2.1 Face-Specific Backbone
- Replace ResNet50 with face-pretrained model
- Options:
  - **EfficientNet-B0** pretrained on ImageNet (lighter, may generalize better)
  - **Vision Transformer (ViT)** pretrained on ImageNet-21k
  - **FaceNet/ArcFace** pretrained on face recognition datasets
- Expected gain: 5-10% accuracy

#### 2.2 Spatial Attention Module
- Add attention layer after backbone
- Learn to focus on eyes, mouth, nose regions
- Can be guided by AU labels (attention on regions with active AUs)
- Expected gain: 3-5% accuracy

#### 2.3 Multi-Scale Feature Fusion
- Extract features at multiple scales
- Combine local (eye region) and global (whole face) features
- Expected gain: 2-4% accuracy

### Phase 3: Advanced Techniques

#### 3.1 AU-Driven Data Augmentation
- Use AU labels to guide augmentation
- Only augment regions that don't affect expression
- Example: If AU12 (smile) is active, preserve mouth region
- Expected gain: 2-3% accuracy

#### 3.2 Curriculum Learning
- Start training with majority classes (easier)
- Gradually introduce minority classes
- Use class difficulty scores based on sample count
- Expected gain: 3-5% accuracy

#### 3.3 Test-Time Augmentation (TTA)
- Apply multiple augmentations at inference
- Average predictions across augmentations
- Expected gain: 1-2% accuracy

#### 3.4 Ensemble Methods
- Train multiple models with different strategies:
  - ResNet50 + multi-task
  - ViT + attention
  - EfficientNet + curriculum
- Combine predictions (voting or weighted average)
- Expected gain: 3-5% accuracy

### Phase 4: Minority Class Specialization

#### 4.1 Few-Shot Learning for Minority Classes
- Treat classes with <50 samples as few-shot
- Use meta-learning approaches
- Expected gain: 5-8% for minority classes

#### 4.2 GAN-Based Data Generation
- Train GAN to generate synthetic samples
- Focus on minority classes
- Expected gain: 3-5% overall

#### 4.3 Self-Supervised Pretraining
- Pretrain on unlabeled face images
- Learn better face representations
- Fine-tune on emotion task
- Expected gain: 5-10% accuracy

## Recommended Implementation Order

### Sprint 1: Multi-Task Learning (Highest Priority)
1. Parse AU labels to binary vectors
2. Implement multi-task training script
3. Train FERMultiTask model
4. Compare with baseline
**Expected: 55% → 60-65%**

### Sprint 2: Architecture Improvements
1. Implement spatial attention module
2. Add multi-scale feature fusion
3. Experiment with different backbones
**Expected: 60-65% → 70-75%**

### Sprint 3: Advanced Training Techniques
1. Implement curriculum learning
2. Add TTA at inference
3. Experiment with ensemble methods
**Expected: 70-75% → 78-82%**

### Sprint 4: Minority Class Focus
1. Implement few-shot learning
2. Add GAN-based data generation
3. Fine-tune with self-supervised pretraining
**Expected: 78-82% → 85%+**

## Key Insights

1. **AU labels are the biggest untapped resource** - they provide explicit supervision about facial muscle movements
2. **Multi-task learning is proven effective** for emotion recognition with AUs
3. **Face-specific features matter more than generic ImageNet features**
4. **Attention mechanisms help** the model focus on relevant facial regions
5. **Minority classes need specialized treatment** - general techniques don't work well

## Questions for User

1. **Priority**: Which phase should we tackle first? (Recommendation: Sprint 1 - Multi-Task Learning)
2. **Compute constraints**: How much GPU memory/time available? (Affects model choice)
3. **Target accuracy**: What's the minimum acceptable accuracy? (Guides how many techniques needed)
4. **Dataset access**: Do we have access to additional face datasets for pretraining?
5. **Time constraints**: How soon do you need results? (Affects implementation strategy)
