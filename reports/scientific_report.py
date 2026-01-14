"""
Scientific Report Generator for Emotion Recognition Project.

This module generates comprehensive scientific reports documenting methodology,
results, and comparisons for the Vision-LLM emotion recognition project.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ScientificReportGenerator:
    """
    Generate comprehensive scientific reports for emotion recognition experiments.
    
    Args:
        output_dir: Directory to save reports
        project_name: Name of the project (default: "Vision-LLM for Compound Emotion Recognition")
    """
    
    def __init__(self, output_dir: str = None, 
                 project_name: str = "Vision-LLM for Compound Emotion Recognition"):
        self.output_dir = output_dir or "."
        self.project_name = project_name
        
        # Report sections
        self.sections = {}
        
        # Create output directory if needed
        os.makedirs(self.output_dir, exist_ok=True)
    
    def add_section(self, section_name: str, content: str):
        """
        Add a section to the report.
        
        Args:
            section_name: Name of the section
            content: Content of the section
        """
        self.sections[section_name] = content
    
    def generate_title_page(self) -> str:
        """Generate title page with project information."""
        title = f"""
# {self.project_name}

## Project Information

**Date:** {datetime.now().strftime("%B %d, %Y")}  
**Authors:** [Your Name Here]  
**Institution:** [Your Institution Here]  

---

## Executive Summary

This report presents the methodology, implementation, and results of a Vision-Language Model (Vision-LLM) approach for compound emotion recognition on the RAF-CE dataset. The project explores the effectiveness of multimodal AI models in classifying and explaining complex facial expressions that combine multiple basic emotions.

### Key Findings

- [Fill in key findings after experiments]

### Contributions

- Implementation of multiple Vision-LLM architectures (BLIP-2, Qwen-VL, InternVL, CLIP)
- Comprehensive benchmarking framework comparing Vision-Only vs Vision-LLM approaches
- Novel textual evaluation metrics for explanation quality (BLEU, ROUGE, CLIPScore, Faithfulness)
- Explainable AI (XAI) integration for model interpretability

---

"""
        return title
    
    def generate_introduction(self) -> str:
        """Generate introduction section."""
        intro = """
## 1. Introduction

### 1.1 Background

Facial Expression Recognition (FER) is a fundamental task in computer vision with applications in human-computer interaction, behavioral psychology, security, social robotics, and mental health. Traditional FER systems have focused on classifying basic emotions (anger, joy, sadness, fear, disgust, surprise) using Convolutional Neural Networks (CNNs).

However, natural human expressions frequently involve **compound emotions** - complex combinations of basic emotions that are challenging to classify discretely. Examples include:
- Happily surprised
- Sadly angry
- Fearfully disgusted
- Happily fearful

These compound expressions involve subtle micro-variations in multiple Action Units (AUs), making them difficult to classify using traditional vision-only approaches.

### 1.2 Motivation

Recent advances in multimodal AI, particularly Vision-Language Models (Vision-LLMs), offer a paradigm shift. By unifying visual perception with linguistic reasoning, Vision-LLMs can:
1. **Classify** compound emotions more accurately by understanding contextual relationships
2. **Explain** predictions through natural language descriptions
3. **Interpret** nuances by linking facial features to emotional states
4. **Justify** classifications with textual evidence

This project leverages Vision-LLMs (BLIP-2, LLaVA, Qwen-VL, InternVL, CLIP) to address the limitations of traditional FER systems and provide interpretable, explainable emotion recognition.

### 1.3 Research Questions

1. **RQ1:** How do Vision-LLM models compare to traditional Vision-Only models in classifying compound emotions on RAF-CE?
2. **RQ2:** Can Vision-LLMs generate accurate and faithful textual explanations for facial expressions?
3. **RQ3:** Which Vision-LLM architecture performs best for compound emotion recognition?
4. **RQ4:** How do textual evaluation metrics (BLEU, ROUGE, CLIPScore) correlate with human judgment of explanation quality?

### 1.4 Objectives

1. Implement a complete pipeline for compound emotion recognition on RAF-CE dataset
2. Develop and benchmark multiple Vision-LLM architectures
3. Create novel textual evaluation metrics for explanation quality
4. Integrate XAI techniques for visual interpretation
5. Compare Vision-Only vs Vision-LLM approaches comprehensively

---

"""
        return intro
    
    def generate_methodology(self) -> str:
        """Generate methodology section."""
        method = """
## 2. Methodology

### 2.1 Dataset

#### RAF-CE Dataset

The **RAF-CE (Real-world Affective Faces - Compound Emotions)** dataset is used for this study.

**Characteristics:**
- **Images:** 12,271 facial images in real-world conditions
- **Classes:** 14 compound emotion categories
- **Annotations:** 
  - Emotion labels (compound expressions)
  - Action Units (AUs) for each image
  - Train/Validation/Test partitions

**Emotion Classes:**
| ID | Emotion | Description |
|-----|-----------|-------------|
| 0 | Happily surprised | Joy + Surprise |
| 1 | Happily disgusted | Joy + Disgust |
| 2 | Sadly fearful | Sadness + Fear |
| 3 | Sadly angry | Sadness + Anger |
| 4 | Sadly surprised | Sadness + Surprise |
| 5 | Sadly disgusted | Sadness + Disgust |
| 6 | Fearfully angry | Fear + Anger |
| 7 | Fearfully surprised | Fear + Surprise |
| 8 | Fearfully disgusted | Fear + Disgust |
| 9 | Angrily surprised | Anger + Surprise |
| 10 | Angrily disgusted | Anger + Disgust |
| 11 | Disgustedly surprised | Disgust + Surprise |
| 12 | Happily fearful | Joy + Fear |
| 13 | Happily sad | Joy + Sadness |

**Partition:**
- Training: 2,709 images (partition_id=0)
- Validation: 931 images (partition_id=2)
- Test: 909 images (partition_id=1)

### 2.2 Pipeline Architecture

The proposed system follows a three-layer architecture:

#### Layer 1: Data Preparation and Preprocessing

**Steps:**
1. **Face Detection and Alignment:** Use aligned images from RAF-CE dataset
2. **Normalization:** Apply ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
3. **Data Augmentation:**
   - Random horizontal flip (p=0.5)
   - Random rotation (±15°)
   - Color jittering (brightness, contrast, saturation ±0.3)
   - Random affine transformations (translation ±10%)
4. **Class Distribution Analysis:** Verify balanced representation across 14 classes

**Rationale:** Augmentation improves generalization and reduces overfitting, especially important for compound emotions with limited samples.

#### Layer 2: Vision-LLM Training

**Model Architectures:**

##### Vision-Only Baselines
1. **ResNet-50:** 23.5M parameters, pretrained on ImageNet
2. **EfficientNet-B0:** 5.3M parameters, compound scaling
3. **Vision Transformer (ViT):** 86.6M parameters, patch-based attention

##### Vision-LLM Models
1. **BLIP-2 (Salesforce/blip2-opt-2.7b):**
   - Vision encoder: ViT-G/14
   - Language model: OPT-2.7B
   - Strengths: Excellent image-text understanding, detailed captioning

2. **LLaVA 1.5-7B (llava-hf/llava-1.5-7b-hf):**
   - Vision encoder: CLIP ViT-L/14
   - Language model: Vicuna-7B
   - Strengths: Strong multimodal reasoning, widely used

3. **Qwen-VL 2.0 (Qwen/Qwen-VL-Chat-7B):**
   - Vision encoder: Qwen-ViT
   - Language model: Qwen-7B
   - Strengths: Excellent for micro-expressions, strong reasoning

4. **InternVL 1.5 (OpenGVLab/InternVL-Chat-V1-5):**
   - Vision encoder: InternViT-6B
   - Language model: InternLM-20B
   - Strengths: Powerful vision capabilities, detailed analysis

5. **CLIP (openai/clip-vit-base-patch32):**
   - Vision encoder: ViT-B/32
   - Text encoder: ViT-B/32
   - Strengths: Zero-shot classification, strong vision-text alignment

**Training Strategy:**
- **LoRA (Low-Rank Adaptation):** r=16, α=32 for efficient fine-tuning
- **4-bit Quantization:** Reduce memory usage for large models
- **Learning Rate:** 1e-4 with cosine annealing warmup
- **Batch Size:** 32 (16 for ViT models)
- **Epochs:** 50 with early stopping

**Prompt Engineering:**

For Vision-LLM explanation generation, we use structured prompts:

```
"Describe the facial expression in this image and identify the compound emotion. 
Explain which facial features (eyebrows, eyes, mouth, Action Units) 
contribute to this expression."
```

For emotion-specific explanations:
```
"This person is feeling [EMOTION_LABEL]. 
Explain why based on their facial features and Action Units."
```

#### Layer 3: Multimodal Interpretation (XAI)

**Visual Interpretation:**
- **Grad-CAM:** Gradient-weighted Class Activation Mapping on vision encoder
- **Attention Maps:** Visualize model attention on facial regions
- **AU Heatmaps:** Highlight Action Units contributing to prediction

**Textual Interpretation:**
- **Explanation Analysis:** Parse generated text for facial region mentions
- **Coherence Check:** Verify alignment between explanation and visual features
- **Faithfulness Score:** Measure correlation between explanation and attention

### 2.3 Evaluation Metrics

#### Classification Metrics
- **Accuracy:** Overall classification correctness
- **F1-Score (Macro):** Harmonic mean of precision and recall (unweighted)
- **F1-Score (Weighted):** Weighted average accounting for class imbalance
- **Confusion Matrix:** Per-class performance analysis
- **Per-Class Metrics:** Precision, Recall, F1 for each of 14 classes

#### Textual Metrics (Novel Contribution)
- **BLEU Score:** Measures n-gram overlap between generated and reference explanations
  - BLEU-1 to BLEU-4 for different n-gram orders
  - Higher values indicate better lexical overlap
  
- **ROUGE Score:** Measures recall-based overlap for summarization quality
  - ROUGE-1, ROUGE-2, ROUGE-L variants
  - Higher values indicate better content coverage
  
- **CLIPScore:** Measures vision-text coherence
  - Cosine similarity between image and generated text
  - Higher values indicate better alignment
  
- **Faithfulness Score:** Measures explanation-attention alignment
  - Average attention on facial regions mentioned in explanation
  - Higher values indicate more faithful explanations

#### Efficiency Metrics
- **Inference Time:** Average time per prediction (ms)
- **Throughput:** Images processed per second
- **Peak Memory:** GPU memory usage during inference (GB)
- **Model Size:** Number of trainable parameters

---

"""
        return method
    
    def generate_results_section(self, benchmark_results: Dict[str, Any]) -> str:
        """Generate results section."""
        results = """
## 3. Results

### 3.1 Model Performance Comparison

#### Classification Performance

| Model | Accuracy | F1-Macro | F1-Weighted | Inference Time (ms) | Throughput (img/s) |
|--------|----------|------------|---------------|---------------------|-------------------|
"""
        
        # Add model results
        for model_name, metrics in benchmark_results.items():
            acc = metrics.get('accuracy', 0) * 100
            f1_macro = metrics.get('f1_macro', 0) * 100
            f1_weighted = metrics.get('f1_weighted', 0) * 100
            time_ms = metrics.get('avg_inference_time_ms', 0)
            throughput = metrics.get('throughput_img_per_sec', 0)
            
            results += f"| {model_name} | {acc:.2f}% | {f1_macro:.2f}% | {f1_weighted:.2f}% | {time_ms:.1f} | {throughput:.1f} |\n"
        
        results += """
#### Best Models by Metric

"""
        
        # Find best models
        metrics_to_check = ['accuracy', 'f1_macro', 'f1_weighted', 
                          'avg_inference_time_ms', 'throughput_img_per_sec']
        metric_names = ['Accuracy', 'F1-Macro', 'F1-Weighted', 
                      'Inference Time', 'Throughput']
        
        for metric, name in zip(metrics_to_check, metric_names):
            best_model = max(benchmark_results.items(), 
                          key=lambda x: x[1].get(metric, 0))
            best_value = best_model[1].get(metric, 0)
            
            if 'time' in metric.lower():
                results += f"- **Best {name} (fastest):** {best_model[0]} ({best_value:.2f} ms)\n"
            else:
                results += f"- **Best {name}:** {best_model[0]} ({best_value:.4f})\n"
        
        # Vision-Only vs Vision-LLM comparison
        results += """
### 3.2 Vision-Only vs Vision-LLM Comparison

"""
        
        vision_only = [k for k in benchmark_results.keys() 
                     if 'ResNet' in k or 'EfficientNet' in k or 'ViT' in k]
        vision_llm = [k for k in benchmark_results.keys() 
                     if 'BLIP' in k or 'Qwen' in k or 'InternVL' in k or 'CLIP' in k]
        
        if vision_only and vision_llm:
            vo_acc = np.mean([benchmark_results[m]['accuracy'] for m in vision_only]) * 100
            vllm_acc = np.mean([benchmark_results[m]['accuracy'] for m in vision_llm]) * 100
            diff = vllm_acc - vo_acc
            
            results += f"**Vision-Only Average Accuracy:** {vo_acc:.2f}%\n"
            results += f"**Vision-LLM Average Accuracy:** {vllm_acc:.2f}%\n"
            results += f"**Improvement:** {diff:+.2f}% ({diff/vo_acc*100:+.1f}%)\n\n"
            
            if diff > 0:
                results += "✓ Vision-LLM models outperform Vision-Only baselines\n"
            else:
                results += "✗ Vision-LLM models underperform Vision-Only baselines\n"
        
        # Textual metrics
        results += """
### 3.3 Textual Explanation Quality

"""
        
        has_textual = any('BLEU' in str(k) or 'ROUGE' in str(k) 
                        for k in benchmark_results.values())
        
        if has_textual:
            results += "| Model | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L | CLIPScore | Faithfulness |\n"
            results += "|--------|---------|---------|---------|---------|-----------|---------------|\n"
            
            for model_name, metrics in benchmark_results.items():
                bleu = metrics.get('BLEU-4', 'N/A')
                rouge1 = metrics.get('ROUGE-1', 'N/A')
                rouge2 = metrics.get('ROUGE-2', 'N/A')
                rougeL = metrics.get('ROUGE-L', 'N/A')
                clipscore = metrics.get('CLIPScore', 'N/A')
                faith = metrics.get('Faithfulness', 'N/A')
                
                results += f"| {model_name} | {bleu} | {rouge1} | {rouge2} | {rougeL} | {clipscore} | {faith} |\n"
        else:
            results += "*Textual metrics not available (explanations not generated)*\n"
        
        return results
    
    def generate_discussion(self, benchmark_results: Dict[str, Any]) -> str:
        """Generate discussion section."""
        discussion = """
## 4. Discussion

### 4.1 Analysis of Results

#### Model Performance

"""
        
        # Analyze top performers
        sorted_models = sorted(benchmark_results.items(), 
                          key=lambda x: x[1].get('accuracy', 0), 
                          reverse=True)
        
        if sorted_models:
            top_model, top_metrics = sorted_models[0]
            discussion += f"**{top_model}** achieved the highest accuracy of **{top_metrics.get('accuracy', 0)*100:.2f}%**. "
            discussion += "This demonstrates the effectiveness of [Vision-LLM / traditional CNN] approaches for compound emotion recognition.\n\n"
        
        # Discuss Vision-LLM advantages
        discussion += """
#### Vision-LLM Advantages

1. **Multimodal Understanding:** Vision-LLMs leverage both visual and linguistic modalities, enabling better understanding of compound emotions that require contextual reasoning.

2. **Explainability:** Unlike black-box CNNs, Vision-LLMs generate natural language explanations that are interpretable by humans.

3. **Zero-Shot Capabilities:** Models like CLIP can classify emotions without fine-tuning, demonstrating strong generalization.

4. **Action Unit Awareness:** Vision-LLMs can identify and explain specific facial muscle movements (AUs) contributing to expressions.

#### Limitations

1. **Computational Cost:** Vision-LLMs require significantly more resources (7B+ parameters) compared to CNNs (23M parameters).

2. **Inference Time:** Larger models have slower inference times (50-200ms vs 5-20ms for CNNs).

3. **Hallucination Risk:** Vision-LLMs may generate explanations that don't perfectly align with visual evidence.

### 4.2 Comparison with Related Work

| Study | Dataset | Best Model | Accuracy | Notes |
|-------|----------|-------------|----------|-------|
| [Related Work 1] | RAF-CE | [X.X]% | [Notes] |
| [Related Work 2] | RAF-CE | [X.X]% | [Notes] |
| **This Study** | RAF-CE | [X.X]% | [Notes] |

*Note: Fill in related work after literature review*

### 4.3 Error Analysis

#### Common Confusions

Based on confusion matrix analysis, the most common misclassifications are:

1. **[Emotion A] ↔ [Emotion B]:** [X]% of cases
   - **Reason:** [Explain why these are confused]
   - **Mitigation:** [Suggest improvements]

2. **[Emotion C] ↔ [Emotion D]:** [X]% of cases
   - **Reason:** [Explain why these are confused]
   - **Mitigation:** [Suggest improvements]

#### Per-Class Performance

"""
        
        # Add per-class analysis if available
        for model_name, metrics in benchmark_results.items():
            if 'per_class_metrics' in metrics:
                per_class = metrics['per_class_metrics']
                worst_class = min(per_class.items(), key=lambda x: x[1]['f1'])
                best_class = max(per_class.items(), key=lambda x: x[1]['f1'])
                
                discussion += f"**{model_name}:**\n"
                discussion += f"- Best performing class: {best_class[0]} (F1={best_class[1]['f1']:.3f})\n"
                discussion += f"- Worst performing class: {worst_class[0]} (F1={worst_class[1]['f1']:.3f})\n\n"
                break
        
        discussion += """
### 4.4 Textual Explanation Quality

The textual evaluation metrics reveal:

"""
        
        # Analyze textual metrics
        has_textual = any('BLEU' in str(k) or 'ROUGE' in str(k) 
                        for k in benchmark_results.values())
        
        if has_textual:
            avg_bleu = np.mean([v.get('BLEU-4', 0) for v in benchmark_results.values() 
                                if 'BLEU-4' in v])
            avg_rouge = np.mean([v.get('ROUGE-1', 0) for v in benchmark_results.values() 
                                 if 'ROUGE-1' in v])
            
            discussion += f"- **Average BLEU-4:** {avg_bleu:.2f}\n"
            discussion += f"- **Average ROUGE-1:** {avg_rouge:.2f}\n\n"
            discussion += "These scores indicate [good/moderate/limited] lexical overlap with reference explanations. "
            discussion += "Improvements could be achieved through [prompt engineering / fine-tuning].\n\n"
        else:
            discussion += "*Textual metrics not computed in this run*\n\n"
        
        discussion += """
### 4.5 Efficiency Considerations

"""
        
        # Analyze efficiency
        avg_time = np.mean([v.get('avg_inference_time_ms', 0) for v in benchmark_results.values()])
        avg_throughput = np.mean([v.get('throughput_img_per_sec', 0) for v in benchmark_results.values()])
        
        discussion += f"- **Average Inference Time:** {avg_time:.1f} ms\n"
        discussion += f"- **Average Throughput:** {avg_throughput:.1f} images/sec\n\n"
        
        if avg_time > 100:
            discussion += "**Real-time Limitations:** Current inference times may be too slow for real-time applications. "
            discussion += "Consider model distillation or quantization for deployment.\n"
        else:
            discussion += "**Real-time Feasibility:** Inference times are suitable for real-time applications.\n"
        
        return discussion
    
    def generate_conclusion(self) -> str:
        """Generate conclusion section."""
        conclusion = """
## 5. Conclusion

### 5.1 Summary of Contributions

This project successfully implemented and evaluated a comprehensive Vision-LLM pipeline for compound emotion recognition on the RAF-CE dataset. Key contributions include:

1. **Complete Pipeline:** End-to-end system from data preprocessing to XAI interpretation
2. **Multiple Architectures:** Implementation and benchmarking of 4 Vision-LLM models (BLIP-2, Qwen-VL, InternVL, CLIP)
3. **Novel Metrics:** Textual evaluation framework (BLEU, ROUGE, CLIPScore, Faithfulness) for explanation quality
4. **Benchmarking Framework:** Systematic comparison of Vision-Only vs Vision-LLM approaches
5. **XAI Integration:** Visual and textual interpretability tools for model understanding

### 5.2 Key Findings

1. **Vision-LLM Performance:** [Fill in after experiments]
2. **Explanation Quality:** [Fill in after experiments]
3. **Efficiency Trade-offs:** [Fill in after experiments]
4. **Best Architecture:** [Fill in after experiments]

### 5.3 Future Work

1. **Model Optimization:** Implement knowledge distillation to compress Vision-LLMs while maintaining performance
2. **Dataset Expansion:** Test on additional compound emotion datasets (e.g., DFEW, FER2013)
3. **Real-time Deployment:** Optimize inference for edge devices and real-time applications
4. **Interactive Refinement:** Develop human-in-the-loop system for continuous improvement
5. **Multimodal Fusion:** Explore audio-visual fusion for more robust emotion recognition

### 5.4 Ethical Considerations

- **Privacy:** Facial recognition systems must respect user privacy and obtain informed consent
- **Bias:** Models should be evaluated for demographic bias (age, gender, ethnicity)
- **Transparency:** Explanations should be honest about model confidence and limitations
- **Application:** Technology should be used to enhance human well-being, not for surveillance or manipulation

---

## References

1. [Add relevant papers after literature review]
2. [Add relevant papers after literature review]
3. [Add relevant papers after literature review]

## Appendix

### A. Model Architectures

[Detailed diagrams of model architectures]

### B. Training Hyperparameters

[Table of hyperparameters for each model]

### C. Additional Results

[Additional tables, figures, and analyses]

---

**Report Generated:** {datetime.now().strftime("%B %d, %Y at %H:%M")}  
**Project:** {self.project_name}

"""
        return conclusion
    
    def generate_full_report(self, benchmark_results: Dict[str, Any] = None) -> str:
        """
        Generate complete scientific report.
        
        Args:
            benchmark_results: Dictionary of benchmark results
        
        Returns:
            Complete report as string
        """
        report = ""
        
        report += self.generate_title_page()
        report += self.generate_introduction()
        report += self.generate_methodology()
        
        if benchmark_results:
            report += self.generate_results_section(benchmark_results)
            report += self.generate_discussion(benchmark_results)
        
        report += self.generate_conclusion()
        
        return report
    
    def save_report(self, report: str, filename: str = None):
        """
        Save report to file.
        
        Args:
            report: Complete report content
            filename: Output filename (default: auto-generated)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'scientific_report_{timestamp}.md'
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ Report saved to: {filepath}")
        return filepath
    
    def generate_figures(self, benchmark_results: Dict[str, Any]) -> List[str]:
        """
        Generate figures for the report.
        
        Args:
            benchmark_results: Dictionary of benchmark results
        
        Returns:
            List of generated figure file paths
        """
        figures = []
        
        # Figure 1: Accuracy Comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        models = list(benchmark_results.keys())
        accuracies = [benchmark_results[m].get('accuracy', 0) * 100 for m in models]
        
        colors = ['#2ecc71' if 'Vision' in m else '#3498db' for m in models]
        bars = ax.bar(models, accuracies, color=colors)
        
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        fig1_path = os.path.join(self.output_dir, 'figure1_accuracy_comparison.png')
        plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
        plt.close()
        figures.append(fig1_path)
        
        # Figure 2: F1 Score Comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        f1_macros = [benchmark_results[m].get('f1_macro', 0) * 100 for m in models]
        f1_weighted = [benchmark_results[m].get('f1_weighted', 0) * 100 for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, f1_macros, width, label='F1-Macro', color='#3498db')
        bars2 = ax.bar(x + width/2, f1_weighted, width, label='F1-Weighted', color='#2ecc71')
        
        ax.set_ylabel('F1 Score (%)', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        fig2_path = os.path.join(self.output_dir, 'figure2_f1_comparison.png')
        plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
        plt.close()
        figures.append(fig2_path)
        
        # Figure 3: Inference Time vs Accuracy
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, metrics in benchmark_results.items():
            accuracy = metrics.get('accuracy', 0) * 100
            time_ms = metrics.get('avg_inference_time_ms', 0)
            
            marker = 'o' if 'Vision' in model_name else 's'
            color = '#e74c3c' if 'Vision' in model_name else '#3498db'
            size = 100 if 'Vision' in model_name else 50
            
            ax.scatter(time_ms, accuracy, marker=marker, s=size, 
                     color=color, alpha=0.7, label=model_name)
        
        ax.set_xlabel('Inference Time (ms)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Accuracy vs Inference Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        fig3_path = os.path.join(self.output_dir, 'figure3_accuracy_vs_time.png')
        plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
        plt.close()
        figures.append(fig3_path)
        
        print(f"✓ Generated {len(figures)} figures")
        return figures


def generate_report_from_benchmark(benchmark_file: str, output_dir: str = None) -> str:
    """
    Generate scientific report from benchmark JSON file.
    
    Args:
        benchmark_file: Path to benchmark results JSON
        output_dir: Output directory for report
    
    Returns:
        Path to generated report
    """
    # Load benchmark results
    with open(benchmark_file, 'r') as f:
        benchmark_results = json.load(f)
    
    # Generate report
    generator = ScientificReportGenerator(output_dir=output_dir)
    report = generator.generate_full_report(benchmark_results)
    
    # Save report
    report_path = generator.save_report(report)
    
    # Generate figures
    figures = generator.generate_figures(benchmark_results)
    
    print(f"\n{'='*60}")
    print("Scientific Report Generated Successfully!")
    print(f"{'='*60}")
    print(f"Report: {report_path}")
    print(f"Figures: {len(figures)} generated")
    
    return report_path


if __name__ == "__main__":
    # Test report generation
    print("Testing Scientific Report Generator...")
    
    # Create dummy benchmark results for testing
    dummy_results = {
        'ResNet50': {
            'accuracy': 0.85,
            'f1_macro': 0.82,
            'f1_weighted': 0.84,
            'avg_inference_time_ms': 15.5,
            'throughput_img_per_sec': 64.5
        },
        'BLIP-2': {
            'accuracy': 0.88,
            'f1_macro': 0.85,
            'f1_weighted': 0.87,
            'avg_inference_time_ms': 125.3,
            'throughput_img_per_sec': 8.0,
            'BLEU-4': 32.5,
            'ROUGE-1': 28.7,
            'ROUGE-2': 15.3,
            'ROUGE-L': 24.1,
            'CLIPScore': 78.5,
            'Faithfulness': 65.2
        },
        'Qwen-VL': {
            'accuracy': 0.89,
            'f1_macro': 0.86,
            'f1_weighted': 0.88,
            'avg_inference_time_ms': 145.2,
            'throughput_img_per_sec': 6.9,
            'BLEU-4': 35.8,
            'ROUGE-1': 31.2,
            'ROUGE-2': 17.5,
            'ROUGE-L': 26.8,
            'CLIPScore': 82.1,
            'Faithfulness': 68.5
        }
    }
    
    # Generate report
    generator = ScientificReportGenerator(output_dir=".")
    report = generator.generate_full_report(dummy_results)
    
    # Save report
    report_path = generator.save_report(report, "test_scientific_report.md")
    
    # Generate figures
    figures = generator.generate_figures(dummy_results)
    
    print(f"\n✓ Test complete!")
    print(f"Report: {report_path}")
    print(f"Figures: {figures}")
