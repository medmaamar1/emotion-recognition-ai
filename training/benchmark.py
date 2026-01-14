"""
Benchmarking Module for Comparing Vision-Only vs Vision-LLM Models.

This module provides comprehensive benchmarking capabilities to compare different
models on the RAF-CE compound emotion recognition task.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from scripts.dataset import RAFCEDataset
from models.baseline import FERBaseline
from models.vision_llm_models import (
    BLIP2Model, QwenVLModel, InternVLModel, CLIPModel, get_model
)
from evaluation.metrics import FEREvaluator
from evaluation.textual_metrics import TextualMetrics, evaluate_explanations


class ModelBenchmark:
    """
    Benchmark different models for compound emotion recognition.
    
    Args:
        models: Dictionary of model name to model instance
        test_loader: DataLoader for test set
        device: Device to run benchmarks on
        output_dir: Directory to save results
    """
    
    def __init__(self, models: Dict[str, torch.nn.Module], 
                 test_loader: DataLoader, device: torch.device,
                 output_dir: str = None):
        self.models = models
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir or get_config()['output_dir']
        
        # Initialize evaluators
        self.emotion_evaluator = FEREvaluator()
        self.textual_evaluator = TextualMetrics(use_bleu=True, use_rouge=True)
        
        # Store results
        self.results = {}
        
        print(f"Initialized benchmark with {len(models)} models")
        for model_name in models.keys():
            print(f"  - {model_name}")
    
    def benchmark_model(self, model_name: str, model: torch.nn.Module,
                     generate_explanations: bool = False) -> Dict[str, any]:
        """
        Benchmark a single model.
        
        Args:
            model_name: Name of the model
            model: Model instance
            generate_explanations: Whether to generate text explanations
        
        Returns:
            Dictionary with benchmark results
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_name}")
        print(f"{'='*60}")
        
        model.eval()
        all_predictions = []
        all_labels = []
        all_explanations = []
        all_references = []
        inference_times = []
        
        # Track memory usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated()
        
        # Run inference
        start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"Testing {model_name}"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Measure inference time
                batch_start = time.time()
                
                # Get predictions
                if hasattr(model, 'classify_emotion'):
                    # Vision-LLM model
                    pred = model.classify_emotion(images[0])
                    emotion_map = {
                        "happily surprised": 0, "happily disgusted": 1, "sadly fearful": 2,
                        "sadly angry": 3, "sadly surprised": 4, "sadly disgusted": 5,
                        "fearfully angry": 6, "fearfully surprised": 7, "fearfully disgusted": 8,
                        "angrily surprised": 9, "angrily disgusted": 10,
                        "disgustedly surprised": 11, "happily fearful": 12, "happily sad": 13
                    }
                    pred_label = emotion_map.get(pred['emotion'], -1)
                    
                    if generate_explanations:
                        exp = model.generate_emotion_explanation(images[0])
                        all_explanations.append(exp)
                        # Create reference (simplified)
                        ref_exp = f"The person shows {pred['emotion']}."
                        all_references.append(ref_exp)
                    
                    batch_time = time.time() - batch_start
                    inference_times.append(batch_time)
                    
                else:
                    # Vision-only model
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    pred_label = predicted.cpu().item()
                    
                    batch_time = time.time() - batch_start
                    inference_times.append(batch_time)
                
                all_predictions.append(pred_label)
                all_labels.append(labels.cpu().item())
        
        total_time = time.time() - start_time
        
        # Get peak memory
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        
        # Calculate metrics
        emotion_metrics = self.emotion_evaluator.calculate_metrics(all_labels, all_predictions)
        
        # Calculate efficiency metrics
        avg_inference_time = np.mean(inference_times) * 1000  # ms
        throughput = len(all_predictions) / total_time  # images/sec
        
        result = {
            'model_name': model_name,
            'total_samples': len(all_predictions),
            'total_time': total_time,
            'avg_inference_time_ms': avg_inference_time,
            'throughput_img_per_sec': throughput,
            'accuracy': emotion_metrics['accuracy'],
            'f1_macro': emotion_metrics['f1_macro'],
            'f1_weighted': emotion_metrics['f1_weighted'],
            'precision_macro': emotion_metrics['precision_macro'],
            'recall_macro': emotion_metrics['recall_macro'],
            'confusion_matrix': emotion_metrics['confusion_matrix'].tolist(),
            'per_class_metrics': emotion_metrics['per_class_metrics']
        }
        
        # Add memory usage
        if torch.cuda.is_available():
            result['peak_memory_gb'] = peak_memory
        
        # Add textual metrics if explanations generated
        if generate_explanations and len(all_explanations) > 0:
            textual_metrics = self.textual_evaluator.compute_all_metrics(
                all_references, all_explanations
            )
            result.update(textual_metrics)
            result['sample_explanations'] = all_explanations[:5]  # First 5 samples
        
        self.results[model_name] = result
        
        # Print summary
        print(f"\nResults for {model_name}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  F1 (Macro): {result['f1_macro']:.4f}")
        print(f"  F1 (Weighted): {result['f1_weighted']:.4f}")
        print(f"  Avg Inference Time: {result['avg_inference_time_ms']:.2f} ms")
        print(f"  Throughput: {result['throughput_img_per_sec']:.2f} img/sec")
        if 'peak_memory_gb' in result:
            print(f"  Peak Memory: {result['peak_memory_gb']:.2f} GB")
        
        return result
    
    def run_full_benchmark(self, generate_explanations: bool = False):
        """
        Run benchmark on all models.
        
        Args:
            generate_explanations: Whether to generate text explanations
        """
        print(f"\n{'='*60}")
        print("Starting Full Benchmark")
        print(f"{'='*60}")
        print(f"Total models: {len(self.models)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        print(f"Generate explanations: {generate_explanations}")
        
        for model_name, model in self.models.items():
            try:
                result = self.benchmark_model(model_name, model, generate_explanations)
            except Exception as e:
                print(f"✗ Error benchmarking {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Generate comparison report
        self.generate_comparison_report()
        
        # Save results
        self.save_results()
    
    def generate_comparison_report(self):
        """Generate comparison report across all models."""
        print(f"\n{'='*60}")
        print("Benchmark Comparison Report")
        print(f"{'='*60}\n")
        
        # Create comparison table
        metrics_to_compare = ['accuracy', 'f1_macro', 'f1_weighted', 
                           'avg_inference_time_ms', 'throughput_img_per_sec']
        
        print(f"{'Model':<20} | {'Accuracy':>10} | {'F1-Macro':>10} | {'F1-Weighted':>12} | {'Time (ms)':>10} | {'Throughput':>10}")
        print("-" * 80)
        
        for model_name, result in self.results.items():
            print(f"{model_name:<20} | {result['accuracy']:>9.2%} | "
                  f"{result['f1_macro']:>9.2%} | {result['f1_weighted']:>10.2%} | "
                  f"{result['avg_inference_time_ms']:>8.1f} | {result['throughput_img_per_sec']:>8.1f}")
        
        # Find best model for each metric
        print(f"\nBest Models by Metric:")
        for metric in metrics_to_compare:
            best_model = max(self.results.items(), 
                          key=lambda x: x[1][metric] if metric in x[1] else 0)
            print(f"  {metric}: {best_model[0]} ({best_model[1].get(metric, 'N/A')})")
        
        # Vision-Only vs Vision-LLM comparison
        vision_only_models = [k for k in self.results.keys() 
                           if 'ResNet' in k or 'ViT' in k or 'EfficientNet' in k]
        vision_llm_models = [k for k in self.results.keys() 
                           if 'BLIP' in k or 'Qwen' in k or 'InternVL' in k or 'CLIP' in k]
        
        if vision_only_models and vision_llm_models:
            print(f"\nVision-Only vs Vision-LLM:")
            vo_avg_acc = np.mean([self.results[m]['accuracy'] for m in vision_only_models])
            vllm_avg_acc = np.mean([self.results[m]['accuracy'] for m in vision_llm_models])
            print(f"  Vision-Only Avg Accuracy: {vo_avg_acc:.2%}")
            print(f"  Vision-LLM Avg Accuracy: {vllm_avg_acc:.2%}")
            print(f"  Difference: {(vllm_avg_acc - vo_avg_acc):+.2%}")
        
        print(f"\n{'='*60}")
    
    def save_results(self):
        """Save benchmark results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f'benchmark_results_{timestamp}.json')
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"\n✓ Results saved to: {results_file}")


def create_baseline_models(device: torch.device, num_classes: int = 14) -> Dict[str, torch.nn.Module]:
    """
    Create baseline vision-only models.
    
    Args:
        device: Device to load models on
        num_classes: Number of emotion classes
    
    Returns:
        Dictionary of model name to model instance
    """
    models = {}
    
    # ResNet50
    print("\nCreating baseline models...")
    models['ResNet50'] = FERBaseline(model_type='resnet50', num_classes=num_classes).to(device)
    
    # EfficientNet-B0
    models['EfficientNet-B0'] = FERBaseline(model_type='efficientnet_b0', num_classes=num_classes).to(device)
    
    # ViT
    models['ViT-Base'] = FERBaseline(model_type='vit_base_patch16_224', num_classes=num_classes).to(device)
    
    print(f"✓ Created {len(models)} baseline models")
    return models


def create_vision_llm_models(device: torch.device, num_classes: int = 14,
                           load_in_4bit: bool = True) -> Dict[str, torch.nn.Module]:
    """
    Create Vision-LLM models.
    
    Args:
        device: Device to load models on
        num_classes: Number of emotion classes
        load_in_4bit: Whether to load in 4-bit (default: True)
    
    Returns:
        Dictionary of model name to model instance
    """
    models = {}
    
    print("\nCreating Vision-LLM models...")
    
    # BLIP-2 (if resources allow)
    try:
        models['BLIP-2'] = BLIP2Model(num_classes=num_classes, 
                                       load_in_4bit=load_in_4bit,
                                       device=device)
    except Exception as e:
        print(f"Warning: Could not load BLIP-2: {e}")
    
    # Qwen-VL (if resources allow)
    try:
        models['Qwen-VL'] = QwenVLModel(num_classes=num_classes,
                                       load_in_4bit=load_in_4bit,
                                       device=device)
    except Exception as e:
        print(f"Warning: Could not load Qwen-VL: {e}")
    
    # InternVL (if resources allow)
    try:
        models['InternVL'] = InternVLModel(num_classes=num_classes,
                                       load_in_4bit=load_in_4bit,
                                       device=device)
    except Exception as e:
        print(f"Warning: Could not load InternVL: {e}")
    
    # CLIP (lightweight)
    try:
        models['CLIP'] = CLIPModel(num_classes=num_classes,
                                   load_in_4bit=False,
                                   device=device)
    except Exception as e:
        print(f"Warning: Could not load CLIP: {e}")
    
    print(f"✓ Created {len(models)} Vision-LLM models")
    return models


def run_benchmark(include_vision_only: bool = True,
                include_vision_llm: bool = True,
                generate_explanations: bool = False,
                batch_size: int = 32):
    """
    Run comprehensive benchmark.
    
    Args:
        include_vision_only: Whether to include baseline models
        include_vision_llm: Whether to include Vision-LLM models
        generate_explanations: Whether to generate text explanations
        batch_size: Batch size for data loading
    """
    # Load configuration
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("Emotion Recognition Model Benchmark")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Generate Explanations: {generate_explanations}")
    print()
    
    # Load test dataset
    from torchvision import transforms
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = RAFCEDataset(partition_id=1, transform=val_transform, use_aligned=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=config['train_config']['num_workers'])
    
    print(f"Test dataset: {len(test_dataset)} samples\n")
    
    # Create models
    all_models = {}
    
    if include_vision_only:
        baseline_models = create_baseline_models(device, config['model_config']['num_classes'])
        all_models.update(baseline_models)
    
    if include_vision_llm:
        vision_llm_models = create_vision_llm_models(device, config['model_config']['num_classes'])
        all_models.update(vision_llm_models)
    
    # Run benchmark
    benchmark = ModelBenchmark(all_models, test_loader, device, config['output_dir'])
    benchmark.run_full_benchmark(generate_explanations=generate_explanations)
    
    print(f"\n{'='*60}")
    print("Benchmark Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark emotion recognition models')
    parser.add_argument('--include_vision_only', action='store_true', default=True,
                       help='Include baseline vision-only models')
    parser.add_argument('--include_vision_llm', action='store_true', default=True,
                       help='Include Vision-LLM models')
    parser.add_argument('--generate_explanations', action='store_true', default=False,
                       help='Generate text explanations')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for data loading')
    
    args = parser.parse_args()
    
    run_benchmark(
        include_vision_only=args.include_vision_only,
        include_vision_llm=args.include_vision_llm,
        generate_explanations=args.generate_explanations,
        batch_size=args.batch_size
    )
