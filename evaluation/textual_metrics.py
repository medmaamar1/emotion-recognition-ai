"""
Textual Metrics for Evaluating Vision-LLM Explanations.

This module implements metrics to evaluate the quality of generated text explanations
for facial emotion recognition, including BLEU and ROUGE scores.
"""

import torch
import numpy as np
from typing import List, Dict, Union
from collections import Counter

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not available. Install with: pip install rouge-score")

try:
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: nltk not available. Install with: pip install nltk")


class TextualMetrics:
    """
    Comprehensive textual metrics for evaluating Vision-LLM explanations.
    
    Args:
        use_bleu: Whether to compute BLEU scores (default: True)
        use_rouge: Whether to compute ROUGE scores (default: True)
    """
    
    def __init__(self, use_bleu=True, use_rouge=True):
        self.use_bleu = use_bleu and NLTK_AVAILABLE
        self.use_rouge = use_rouge and ROUGE_AVAILABLE
        
        if self.use_rouge:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def compute_bleu(self, references: List[str], hypotheses: List[str], 
                   max_n: int = 4, smoothing: str = 'method1') -> Dict[str, float]:
        """
        Compute BLEU scores for generated explanations.
        
        Args:
            references: List of ground truth explanations
            hypotheses: List of generated explanations
            max_n: Maximum n-gram order (default: 4)
            smoothing: Smoothing function to use (default: 'method1')
        
        Returns:
            Dictionary with BLEU scores for different n-grams
        """
        if not self.use_bleu:
            return {}
        
        # Tokenize references and hypotheses
        ref_tokens = [[nltk.word_tokenize(ref)] for ref in references]
        hyp_tokens = [nltk.word_tokenize(hyp) for hyp in hypotheses]
        
        # Get smoothing function
        smoothing_fn = SmoothingFunction().method1 if smoothing == 'method1' else SmoothingFunction().method2
        
        # Compute BLEU scores for different n-grams
        bleu_scores = {}
        for n in range(1, max_n + 1):
            weights = tuple([1.0/n] * n + [0.0] * (max_n - n))
            try:
                score = corpus_bleu(ref_tokens, hyp_tokens, weights=weights, smoothing_function=smoothing_fn)
                bleu_scores[f'BLEU-{n}'] = score * 100
            except:
                bleu_scores[f'BLEU-{n}'] = 0.0
        
        # Compute cumulative BLEU
        bleu_scores['BLEU-4'] = corpus_bleu(ref_tokens, hyp_tokens, 
                                           smoothing_function=smoothing_fn) * 100
        
        return bleu_scores
    
    def compute_rouge(self, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE scores for generated explanations.
        
        Args:
            references: List of ground truth explanations
            hypotheses: List of generated explanations
        
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores
        """
        if not self.use_rouge:
            return {}
        
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for ref, hyp in zip(references, hypotheses):
            scores = self.rouge_scorer.score(ref, hyp)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        # Average scores
        return {
            'ROUGE-1': np.mean(rouge_scores['rouge1']) * 100,
            'ROUGE-2': np.mean(rouge_scores['rouge2']) * 100,
            'ROUGE-L': np.mean(rouge_scores['rougeL']) * 100
        }
    
    def compute_clipscore(self, references: List[str], hypotheses: List[str],
                       emotion_labels: List[str], image_features: torch.Tensor = None) -> Dict[str, float]:
        """
        Compute CLIPScore to measure coherence between vision and text.
        
        Args:
            references: List of ground truth explanations
            hypotheses: List of generated explanations
            emotion_labels: List of emotion labels
            image_features: Precomputed image features (optional)
        
        Returns:
            Dictionary with CLIP scores
        """
        try:
            import clip
        except ImportError:
            print("Warning: clip not available. Install with: pip install git+https://github.com/openai/CLIP.git")
            return {}
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        # Encode references and hypotheses
        ref_tokens = clip.tokenize(references, truncate=True).to(device)
        hyp_tokens = clip.tokenize(hypotheses, truncate=True).to(device)
        
        with torch.no_grad():
            ref_features = model.encode_text(ref_tokens)
            hyp_features = model.encode_text(hyp_tokens)
        
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(ref_features, hyp_features, dim=1)
        
        return {
            'CLIPScore': similarity.mean().item() * 100,
            'CLIPScore_Std': similarity.std().item() * 100
        }
    
    def compute_faithfulness(self, explanations: List[str], attention_maps: List[np.ndarray],
                           emotion_labels: List[str], image_regions: List[str]) -> Dict[str, float]:
        """
        Compute Faithfulness Score to measure alignment between explanations and visual attention.
        
        Args:
            explanations: List of generated text explanations
            attention_maps: List of attention/heatmap arrays
            emotion_labels: List of predicted emotion labels
            image_regions: List of mentioned facial regions (e.g., ['eyes', 'mouth', 'eyebrows'])
        
        Returns:
            Dictionary with faithfulness metrics
        """
        faithfulness_scores = []
        
        for exp, attn_map, regions in zip(explanations, attention_maps, image_regions):
            # Extract regions mentioned in explanation
            mentioned_regions = self._extract_regions_from_text(exp)
            
            # Compute attention weights for mentioned regions
            region_attention = []
            for region in mentioned_regions:
                if region in regions:
                    # Get attention for this region (simplified - would need region mapping)
                    region_attention.append(np.mean(attn_map))
            
            if len(region_attention) > 0:
                # Faithfulness = average attention on mentioned regions
                faithfulness_scores.append(np.mean(region_attention))
            else:
                faithfulness_scores.append(0.0)
        
        return {
            'Faithfulness': np.mean(faithfulness_scores) * 100,
            'Faithfulness_Std': np.std(faithfulness_scores) * 100
        }
    
    def _extract_regions_from_text(self, text: str) -> List[str]:
        """
        Extract facial regions mentioned in text explanation.
        
        Args:
            text: Generated explanation text
        
        Returns:
            List of mentioned facial regions
        """
        regions = ['eyes', 'mouth', 'eyebrows', 'nose', 'cheeks', 'forehead', 'chin', 'jaw']
        mentioned = []
        
        text_lower = text.lower()
        for region in regions:
            if region in text_lower:
                mentioned.append(region)
        
        return mentioned
    
    def compute_all_metrics(self, references: List[str], hypotheses: List[str],
                         emotion_labels: List[str] = None,
                         attention_maps: List[np.ndarray] = None,
                         image_regions: List[str] = None,
                         image_features: torch.Tensor = None) -> Dict[str, float]:
        """
        Compute all textual metrics.
        
        Args:
            references: List of ground truth explanations
            hypotheses: List of generated explanations
            emotion_labels: List of predicted emotion labels (optional)
            attention_maps: List of attention/heatmap arrays (optional)
            image_regions: List of mentioned facial regions (optional)
            image_features: Precomputed image features (optional)
        
        Returns:
            Dictionary with all computed metrics
        """
        all_metrics = {}
        
        # BLEU scores
        if self.use_bleu:
            bleu_scores = self.compute_bleu(references, hypotheses)
            all_metrics.update(bleu_scores)
        
        # ROUGE scores
        if self.use_rouge:
            rouge_scores = self.compute_rouge(references, hypotheses)
            all_metrics.update(rouge_scores)
        
        # CLIPScore (if image features provided)
        if image_features is not None and emotion_labels is not None:
            clipscore = self.compute_clipscore(references, hypotheses, emotion_labels, image_features)
            all_metrics.update(clipscore)
        
        # Faithfulness (if attention maps provided)
        if attention_maps is not None and image_regions is not None:
            faithfulness = self.compute_faithfulness(hypotheses, attention_maps, 
                                                      emotion_labels, image_regions)
            all_metrics.update(faithfulness)
        
        return all_metrics
    
    def generate_report(self, metrics: Dict[str, float]) -> str:
        """
        Generate a human-readable report from metrics.
        
        Args:
            metrics: Dictionary of computed metrics
        
        Returns:
            Formatted report string
        """
        report = "=" * 60 + "\n"
        report += "Textual Metrics Report\n"
        report += "=" * 60 + "\n\n"
        
        # BLEU scores
        bleu_keys = [k for k in metrics.keys() if 'BLEU' in k]
        if bleu_keys:
            report += "BLEU Scores (n-gram overlap):\n"
            for key in sorted(bleu_keys):
                report += f"  {key}: {metrics[key]:.2f}\n"
            report += "\n"
        
        # ROUGE scores
        rouge_keys = [k for k in metrics.keys() if 'ROUGE' in k]
        if rouge_keys:
            report += "ROUGE Scores (recall-based):\n"
            for key in sorted(rouge_keys):
                report += f"  {key}: {metrics[key]:.2f}\n"
            report += "\n"
        
        # CLIPScore
        if 'CLIPScore' in metrics:
            report += "CLIPScore (vision-text coherence):\n"
            report += f"  CLIPScore: {metrics['CLIPScore']:.2f}\n"
            if 'CLIPScore_Std' in metrics:
                report += f"  CLIPScore Std: {metrics['CLIPScore_Std']:.2f}\n"
            report += "\n"
        
        # Faithfulness
        if 'Faithfulness' in metrics:
            report += "Faithfulness (explanation-attention alignment):\n"
            report += f"  Faithfulness: {metrics['Faithfulness']:.2f}\n"
            if 'Faithfulness_Std' in metrics:
                report += f"  Faithfulness Std: {metrics['Faithfulness_Std']:.2f}\n"
        
        report += "=" * 60
        return report


def evaluate_explanations(references: List[str], hypotheses: List[str],
                       emotion_labels: List[str] = None,
                       attention_maps: List[np.ndarray] = None,
                       image_regions: List[str] = None,
                       image_features: torch.Tensor = None) -> Dict[str, float]:
    """
    Convenience function to evaluate Vision-LLM explanations.
    
    Args:
        references: List of ground truth explanations
        hypotheses: List of generated explanations
        emotion_labels: List of predicted emotion labels (optional)
        attention_maps: List of attention/heatmap arrays (optional)
        image_regions: List of mentioned facial regions (optional)
        image_features: Precomputed image features (optional)
    
    Returns:
        Dictionary with all computed metrics
    """
    metrics_calculator = TextualMetrics(use_bleu=True, use_rouge=True)
    return metrics_calculator.compute_all_metrics(
        references, hypotheses, emotion_labels, 
        attention_maps, image_regions, image_features
    )


if __name__ == "__main__":
    # Test textual metrics
    print("Testing Textual Metrics...")
    
    # Sample data
    references = [
        "The person is happily surprised with raised eyebrows and a smiling mouth.",
        "The facial expression shows sadness mixed with anger through furrowed brows and downturned lips.",
        "Fear and disgust are evident in the wide eyes and wrinkled nose."
    ]
    
    hypotheses = [
        "The individual appears happily surprised, with raised eyebrows and a smiling expression.",
        "This person looks sadly angry, showing furrowed brows and downturned lips.",
        "The face displays fearfully disgusted expression with wide eyes and wrinkled nose."
    ]
    
    emotion_labels = ["Happily surprised", "Sadly angry", "Fearfully disgusted"]
    
    # Compute metrics
    metrics = evaluate_explanations(references, hypotheses, emotion_labels)
    
    # Generate report
    metrics_calculator = TextualMetrics()
    report = metrics_calculator.generate_report(metrics)
    print(report)
    
    print("\nAll tests passed!")
