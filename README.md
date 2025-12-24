# FER-CE: Facial Emotion Recognition of Compound Expressions

This research project focuses on using Vision-LLMs (LLaVA-1.5) to recognize and explain complex human emotions, benchmarked against traditional Vision models (ResNet/ViT).

## 🚀 Getting Started on Kaggle

Since this project requires GPU acceleration (especially for the Vision-LLM), it is recommended to run experiments on Kaggle.

### 1. Upload the Dataset
Upload the RAF-CE dataset as a private Kaggle Dataset. Structure should be:
- `RAF-AU/aligned/` (images)
- `RAFCE_emolabel.txt`
- `RAFCE_partition.txt`
- `RAFCE_AUlabel.txt`

### 2. Environment Setup
In a Kaggle Notebook, run:
```python
!git clone https://github.com/YOUR_USERNAME/emotion-recognition-ai.git
%cd emotion-recognition-ai
!pip install -r requirements.txt
```

### 3. Training Baselines
```python
!python training/train_baseline.py
```

### 4. Vision-LLM Fine-tuning
Ensure GPU T4 x2 or P100 is selected.
```python
!python training/train_vllm.py
```

## 📁 Repository Structure
- `app/`: Streamlit Demonstration Dashboard.
- `evaluation/`: Metrics (F1, Confusion Matrix) and XAI (Grad-CAM).
- `models/`: ResNet, ViT, and LLaVA architectures.
- `scripts/`: Data loading and distribution analysis.
- `training/`: Specialized training loops for each model type.

## 🧪 XAI & Interpretability
The project utilizes Grad-CAM to visualize facial regions contributing to compound emotion predictions, providing a bridge between AU activation and emotional labels.
