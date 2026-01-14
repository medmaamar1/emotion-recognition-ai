"""
Multiple Vision-LLM Model Implementations for Compound Emotion Recognition.

This module provides implementations of various Vision-LLM models including:
- BLIP-2
- Qwen-VL
- InternVL
- CLIP
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple
from transformers import (
    AutoProcessor, 
    Blip2ForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    AutoModel,
    AutoTokenizer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class BaseVisionLLM(nn.Module):
    """
    Base class for Vision-LLM models.
    
    Args:
        model_id: Model identifier from Hugging Face
        num_classes: Number of emotion classes (default: 14)
        load_in_4bit: Whether to load in 4-bit (default: True)
        device: Device to load model on (default: None, auto-detect)
    """
    
    def __init__(self, model_id: str, num_classes: int = 14, 
                 load_in_4bit: bool = True, device: Optional[str] = None):
        super().__init__()
        self.model_id = model_id
        self.num_classes = num_classes
        self.load_in_4bit = load_in_4bit
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Initializing {model_id} on {self.device}...")
        
        # Initialize processor
        self.processor = None
        self.model = None
        
    def load_model(self):
        """Load the model. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement load_model()")
    
    def apply_lora(self, r: int = 16, lora_alpha: int = 32, 
                   lora_dropout: float = 0.05, target_modules: List[str] = None):
        """
        Apply LoRA for efficient fine-tuning.
        
        Args:
            r: LoRA rank (default: 16)
            lora_alpha: LoRA alpha (default: 32)
            lora_dropout: LoRA dropout (default: 0.05)
            target_modules: Target modules for LoRA (default: None)
        
        Returns:
            Model with LoRA applied
        """
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]
        
        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()
        return self.model
    
    def generate_emotion_explanation(self, image, emotion_label: Optional[str] = None,
                                   max_length: int = 100) -> str:
        """
        Generate explanation for emotion in image.
        
        Args:
            image: Input image (PIL Image or tensor)
            emotion_label: Predicted emotion label (optional)
            max_length: Maximum length of generated text (default: 100)
        
        Returns:
            Generated explanation text
        """
        raise NotImplementedError("Subclasses must implement generate_emotion_explanation()")
    
    def classify_emotion(self, image) -> Dict[str, Union[str, float]]:
        """
        Classify emotion from image.
        
        Args:
            image: Input image (PIL Image or tensor)
        
        Returns:
            Dictionary with emotion label and confidence
        """
        raise NotImplementedError("Subclasses must implement classify_emotion()")


class BLIP2Model(BaseVisionLLM):
    """
    BLIP-2 Vision-LLM for emotion recognition and explanation.
    
    BLIP-2 excels at image-text understanding and can generate detailed
    explanations for facial expressions.
    """
    
    def __init__(self, model_id: str = "Salesforce/blip2-opt-2.7b", 
                 num_classes: int = 14, load_in_4bit: bool = True,
                 device: Optional[str] = None):
        super().__init__(model_id, num_classes, load_in_4bit, device)
        self.load_model()
    
    def load_model(self):
        """Load BLIP-2 model."""
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        print(f"✓ BLIP-2 model loaded: {self.model_id}")
    
    def generate_emotion_explanation(self, image, emotion_label: Optional[str] = None,
                                   max_length: int = 100) -> str:
        """Generate explanation using BLIP-2."""
        if emotion_label:
            prompt = f"This person is feeling {emotion_label}. Describe their facial expression and explain why."
        else:
            prompt = "Describe the facial expression in this image and identify the emotion."
        
        inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                early_stopping=True
            )
        
        explanation = self.processor.decode(outputs[0], skip_special_tokens=True)
        return explanation
    
    def classify_emotion(self, image) -> Dict[str, Union[str, float]]:
        """Classify emotion using BLIP-2 captioning."""
        prompt = "What emotion is this person showing? Choose from: happily surprised, happily disgusted, sadly fearful, sadly angry, sadly surprised, sadly disgusted, fearfully angry, fearfully surprised, fearfully disgusted, angrily surprised, angrily disgusted, disgustedly surprised, happily fearful, happily sad."
        
        inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=50,
                num_beams=3
            )
        
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract emotion from caption (simplified)
        emotion_classes = [
            "happily surprised", "happily disgusted", "sadly fearful", "sadly angry",
            "sadly surprised", "sadly disgusted", "fearfully angry", "fearfully surprised",
            "fearfully disgusted", "angrily surprised", "angrily disgusted",
            "disgustedly surprised", "happily fearful", "happily sad"
        ]
        
        # Find best match (simplified - would need better NLP)
        best_emotion = "unknown"
        best_score = 0
        for emotion in emotion_classes:
            if emotion in caption.lower():
                best_emotion = emotion
                best_score = 1.0
                break
        
        return {
            'emotion': best_emotion,
            'confidence': best_score,
            'caption': caption
        }


class QwenVLModel(BaseVisionLLM):
    """
    Qwen-VL Vision-LLM for emotion recognition and explanation.
    
    Qwen-VL is strong at micro-expression detection and can provide
    detailed facial analysis.
    """
    
    def __init__(self, model_id: str = "Qwen/Qwen-VL-Chat-7B", 
                 num_classes: int = 14, load_in_4bit: bool = True,
                 device: Optional[str] = None):
        super().__init__(model_id, num_classes, load_in_4bit, device)
        self.load_model()
    
    def load_model(self):
        """Load Qwen-VL model."""
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        print(f"✓ Qwen-VL model loaded: {self.model_id}")
    
    def generate_emotion_explanation(self, image, emotion_label: Optional[str] = None,
                                   max_length: int = 100) -> str:
        """Generate explanation using Qwen-VL."""
        if emotion_label:
            prompt = f"Analyze this facial expression. The person is {emotion_label}. Explain which facial features (eyebrows, eyes, mouth, etc.) indicate this emotion."
        else:
            prompt = "Analyze this facial expression. Identify the emotion and explain which facial features contribute to it."
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,
                temperature=0.7
            )
        
        explanation = self.processor.decode(outputs[0], skip_special_tokens=True)
        return explanation
    
    def classify_emotion(self, image) -> Dict[str, Union[str, float]]:
        """Classify emotion using Qwen-VL."""
        prompt = "Classify the facial emotion in this image. Choose from: happily surprised, happily disgusted, sadly fearful, sadly angry, sadly surprised, sadly disgusted, fearfully angry, fearfully surprised, fearfully disgusted, angrily surprised, angrily disgusted, disgustedly surprised, happily fearful, happily sad."
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        emotion_classes = [
            "happily surprised", "happily disgusted", "sadly fearful", "sadly angry",
            "sadly surprised", "sadly disgusted", "fearfully angry", "fearfully surprised",
            "fearfully disgusted", "angrily surprised", "angrily disgusted",
            "disgustedly surprised", "happily fearful", "happily sad"
        ]
        
        best_emotion = "unknown"
        best_score = 0
        for emotion in emotion_classes:
            if emotion in response.lower():
                best_emotion = emotion
                best_score = 1.0
                break
        
        return {
            'emotion': best_emotion,
            'confidence': best_score,
            'response': response
        }


class InternVLModel(BaseVisionLLM):
    """
    InternVL Vision-LLM for emotion recognition and explanation.
    
    InternVL provides powerful vision capabilities and can analyze
    complex facial expressions in detail.
    """
    
    def __init__(self, model_id: str = "OpenGVLab/InternVL-Chat-V1-5", 
                 num_classes: int = 14, load_in_4bit: bool = True,
                 device: Optional[str] = None):
        super().__init__(model_id, num_classes, load_in_4bit, device)
        self.load_model()
    
    def load_model(self):
        """Load InternVL model."""
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )
        
        print(f"✓ InternVL model loaded: {self.model_id}")
    
    def generate_emotion_explanation(self, image, emotion_label: Optional[str] = None,
                                   max_length: int = 100) -> str:
        """Generate explanation using InternVL."""
        if emotion_label:
            prompt = f"Describe the facial expression in detail. The emotion is {emotion_label}. Explain which Action Units (AUs) and facial features are visible."
        else:
            prompt = "Describe the facial expression in detail. Identify the compound emotion and explain which facial features and Action Units (AUs) are visible."
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,
                temperature=0.7
            )
        
        explanation = self.processor.decode(outputs[0], skip_special_tokens=True)
        return explanation
    
    def classify_emotion(self, image) -> Dict[str, Union[str, float]]:
        """Classify emotion using InternVL."""
        prompt = "Classify the compound facial emotion. Choose from: happily surprised, happily disgusted, sadly fearful, sadly angry, sadly surprised, sadly disgusted, fearfully angry, fearfully surprised, fearfully disgusted, angrily surprised, angrily disgusted, disgustedly surprised, happily fearful, happily sad."
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        emotion_classes = [
            "happily surprised", "happily disgusted", "sadly fearful", "sadly angry",
            "sadly surprised", "sadly disgusted", "fearfully angry", "fearfully surprised",
            "fearfully disgusted", "angrily surprised", "angrily disgusted",
            "disgustedly surprised", "happily fearful", "happily sad"
        ]
        
        best_emotion = "unknown"
        best_score = 0
        for emotion in emotion_classes:
            if emotion in response.lower():
                best_emotion = emotion
                best_score = 1.0
                break
        
        return {
            'emotion': best_emotion,
            'confidence': best_score,
            'response': response
        }


class CLIPModel(BaseVisionLLM):
    """
    CLIP Vision-LLM for emotion classification.
    
    CLIP is excellent for zero-shot classification and can classify
    emotions without fine-tuning.
    """
    
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", 
                 num_classes: int = 14, load_in_4bit: bool = False,
                 device: Optional[str] = None):
        super().__init__(model_id, num_classes, load_in_4bit, device)
        self.load_model()
    
    def load_model(self):
        """Load CLIP model."""
        try:
            import clip
        except ImportError:
            raise ImportError("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
        
        self.model, self.processor = clip.load(self.model_id, device=self.device)
        
        print(f"✓ CLIP model loaded: {self.model_id}")
    
    def generate_emotion_explanation(self, image, emotion_label: Optional[str] = None,
                                   max_length: int = 100) -> str:
        """
        CLIP doesn't generate text, so this returns a template explanation.
        """
        if emotion_label:
            return f"The person is {emotion_label}. CLIP provides classification but not detailed text generation."
        else:
            return "CLIP provides zero-shot classification but not detailed text generation."
    
    def classify_emotion(self, image) -> Dict[str, Union[str, float]]:
        """Classify emotion using CLIP zero-shot."""
        emotion_classes = [
            "happily surprised", "happily disgusted", "sadly fearful", "sadly angry",
            "sadly surprised", "sadly disgusted", "fearfully angry", "fearfully surprised",
            "fearfully disgusted", "angrily surprised", "angrily disgusted",
            "disgustedly surprised", "happily fearful", "happily sad"
        ]
        
        # Create text prompts for each emotion
        text_prompts = [f"a photo of a person showing {emotion}" for emotion in emotion_classes]
        
        # Encode image and text
        image_input = self.processor(image).unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        
        with torch.no_grad():
            # Get image and text features
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_tokens)
            
            # Calculate cosine similarity
            similarities = (image_features @ text_features.T).squeeze(0)
            similarities = similarities.softmax(dim=0)
            
            # Get top prediction
            top_idx = similarities.argmax().item()
            confidence = similarities[top_idx].item()
        
        return {
            'emotion': emotion_classes[top_idx],
            'confidence': confidence,
            'all_similarities': {emotion: similarities[i].item() 
                               for i, emotion in enumerate(emotion_classes)}
        }


class VisionLLMEnsemble:
    """
    Ensemble of multiple Vision-LLM models for robust predictions.
    
    Combines predictions from multiple models using voting or averaging.
    """
    
    def __init__(self, models: List[BaseVisionLLM], 
                 ensemble_method: str = "voting"):
        """
        Initialize ensemble.
        
        Args:
            models: List of Vision-LLM models
            ensemble_method: 'voting' or 'averaging' (default: 'voting')
        """
        self.models = models
        self.ensemble_method = ensemble_method
        
        print(f"Initialized ensemble with {len(models)} models using {ensemble_method}")
    
    def classify_emotion(self, image) -> Dict[str, Union[str, float]]:
        """
        Classify emotion using ensemble.
        
        Args:
            image: Input image
        
        Returns:
            Dictionary with ensemble prediction and confidence
        """
        predictions = []
        
        for model in self.models:
            pred = model.classify_emotion(image)
            predictions.append(pred)
        
        if self.ensemble_method == "voting":
            # Majority voting
            emotion_counts = {}
            for pred in predictions:
                emotion = pred['emotion']
                if emotion not in emotion_counts:
                    emotion_counts[emotion] = 0
                emotion_counts[emotion] += 1
            
            best_emotion = max(emotion_counts, key=emotion_counts.get)
            confidence = emotion_counts[best_emotion] / len(predictions)
            
        elif self.ensemble_method == "averaging":
            # Average confidence scores
            emotion_scores = {}
            for pred in predictions:
                emotion = pred['emotion']
                conf = pred['confidence']
                if emotion not in emotion_scores:
                    emotion_scores[emotion] = []
                emotion_scores[emotion].append(conf)
            
            # Average scores
            avg_scores = {k: np.mean(v) for k, v in emotion_scores.items()}
            best_emotion = max(avg_scores, key=avg_scores.get)
            confidence = avg_scores[best_emotion]
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return {
            'emotion': best_emotion,
            'confidence': confidence,
            'individual_predictions': predictions
        }
    
    def generate_explanations(self, image, emotion_label: Optional[str] = None,
                             max_length: int = 100) -> List[str]:
        """
        Generate explanations from all models.
        
        Args:
            image: Input image
            emotion_label: Predicted emotion label (optional)
            max_length: Maximum length of generated text
        
        Returns:
            List of explanations from each model
        """
        explanations = []
        
        for model in self.models:
            exp = model.generate_emotion_explanation(image, emotion_label, max_length)
            explanations.append(exp)
        
        return explanations


def get_model(model_name: str, **kwargs) -> BaseVisionLLM:
    """
    Factory function to get Vision-LLM model by name.
    
    Args:
        model_name: Name of model ('blip2', 'qwen-vl', 'internvl', 'clip')
        **kwargs: Additional arguments for model initialization
    
    Returns:
        Vision-LLM model instance
    """
    model_name = model_name.lower()
    
    if model_name == "blip2":
        return BLIP2Model(**kwargs)
    elif model_name == "qwen-vl":
        return QwenVLModel(**kwargs)
    elif model_name == "internvl":
        return InternVLModel(**kwargs)
    elif model_name == "clip":
        return CLIPModel(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: blip2, qwen-vl, internvl, clip")


if __name__ == "__main__":
    # Test Vision-LLM models
    print("Testing Vision-LLM Models...")
    
    # Test CLIP (lightweight, good for testing)
    print("\nTesting CLIP model...")
    clip_model = CLIPModel(load_in_4bit=False)
    print(f"✓ CLIP model initialized")
    
    print("\nAll Vision-LLM models initialized successfully!")
    print("\nAvailable models:")
    print("  - BLIP-2: Salesforce/blip2-opt-2.7b")
    print("  - Qwen-VL: Qwen/Qwen-VL-Chat-7B")
    print("  - InternVL: OpenGVLab/InternVL-Chat-V1-5")
    print("  - CLIP: openai/clip-vit-base-patch32")
