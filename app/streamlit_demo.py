"""
Streamlit Demo Interface for Emotion Recognition.

This provides an interactive web interface to upload images,
classify compound emotions, and get explanations from Vision-LLM models.
"""

import os
import sys
import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import get_config
from models.vision_llm_models import get_model, VisionLLMEnsemble
from evaluation.xai_gradcam import generate_gradcam
from evaluation.textual_metrics import TextualMetrics

# Page configuration
st.set_page_config(
    page_title="Vision-LLM Compound Emotion Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def load_model_page():
    """Model loading page."""
    st.markdown('<div class="main-header"><h1>🎭 Vision-LLM Compound Emotion Recognition</h1></div>', 
                 unsafe_allow_html=True)
    
    st.markdown("""
    ## Welcome!
    
    This demo showcases a Vision-Language Model (Vision-LLM) approach for recognizing 
    **compound emotions** in facial expressions. Unlike traditional systems that classify basic emotions,
    our approach can understand and explain complex emotional states.
    
    ### Features
    - 🎯 **Compound Emotion Classification:** 14 emotion categories
    - 📝 **Natural Language Explanations:** Understandable by humans
    - 🔍 **Visual Interpretation:** XAI heatmaps and attention maps
    - 🤖 **Multiple Models:** BLIP-2, Qwen-VL, InternVL, CLIP
    - 📊 **Comprehensive Metrics:** Accuracy, F1, BLEU, ROUGE
    """)
    
    st.info("👆 Use the sidebar to configure and run the demo!")


def main():
    """Main application."""
    # Page navigation
    page = st.sidebar.radio("Navigation", ["🏠 Home", "🖼️ Upload & Classify", "📊 Benchmark Results", "📖 About"])
    
    if page == "🏠 Home":
        load_model_page()
    
    elif page == "🖼️ Upload & Classify":
        upload_classify_page()
    
    elif page == "📊 Benchmark Results":
        benchmark_results_page()
    
    elif page == "📖 About":
        about_page()


def upload_classify_page():
    """Upload and classify page."""
    st.header("🖼️ Upload & Classify")
    
    # Sidebar configuration
    st.sidebar.subheader("⚙️ Configuration")
    
    # Model selection
    model_options = ["BLIP-2", "Qwen-VL", "InternVL", "CLIP", "Ensemble (All)"]
    selected_model = st.sidebar.selectbox(
        "Select Vision-LLM Model",
        model_options,
        index=0,
        help="Choose which Vision-LLM model to use for classification"
    )
    
    # Advanced options
    st.sidebar.subheader("🔧 Advanced Options")
    generate_explanation = st.sidebar.checkbox("Generate Explanation", value=True)
    show_xai = st.sidebar.checkbox("Show XAI Heatmap", value=True)
    use_ensemble = st.sidebar.checkbox("Use Ensemble (All Models)", value=False)
    
    # Load model
    @st.cache_resource
    def load_vision_llm(model_name, ensemble=False):
        try:
            config = get_config()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            if ensemble:
                # Load multiple models
                models = {}
                for m in ["BLIP-2", "Qwen-VL", "InternVL", "CLIP"]:
                    try:
                        models[m] = get_model(m.lower(), num_classes=14, 
                                               load_in_4bit=False, device=device)
                    except:
                        pass
                return VisionLLMEnsemble(list(models.values()), ensemble_method="voting")
            else:
                # Load single model
                return get_model(model_name.lower(), num_classes=14, 
                                     load_in_4bit=False, device=device)
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_vision_llm(selected_model, use_ensemble)
    
    if model is None:
        st.warning("⚠️ Model not loaded. Please try a different model or check your setup.")
        return
    
    st.success(f"✅ {selected_model} loaded successfully!")
    
    # File upload
    st.subheader("📤 Upload Image")
    
    # Upload options
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            help="Upload a facial image for emotion recognition"
        )
    
    with col2:
        sample_images = st.button("📷 Use Sample Image")
    
    # Load image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    elif sample_images:
        # Use a sample image from dataset
        try:
            from scripts.dataset import RAFCEDataset
            dataset = RAFCEDataset(partition_id=1, use_aligned=True)
            sample = dataset[0]
            image = sample['image']
            st.image(image, caption=f"Sample: {sample['image_id']}", use_column_width=True)
        except:
            st.warning("⚠️ Could not load sample image. Please upload your own.")
            image = None
    else:
        st.info("👆 Please upload an image to begin classification.")
        image = None
    
    # Classify button
    if image is not None:
        if st.button("🎯 Classify Emotion", type="primary", use_container_width=True):
            classify_image(image, model, generate_explanation, show_xai)


def classify_image(image, model, generate_explanation, show_xai):
    """Classify image and display results."""
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔄 Processing image...")
    progress_bar.progress(20)
    
    try:
        # Classify emotion
        progress_bar.progress(40)
        status_text.text("🔍 Classifying emotion...")
        
        result = model.classify_emotion(image)
        
        progress_bar.progress(60)
        
        if result is None:
            st.error("❌ Classification failed. Please try again.")
            return
        
        # Display results
        st.markdown("---")
        st.subheader("🎯 Classification Results")
        
        # Create columns for results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Predicted Emotion")
            st.markdown(f"## {result['emotion']}")
            
            if 'confidence' in result:
                confidence = result['confidence'] * 100
                st.metric("Confidence", f"{confidence:.1f}%")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Emotion description
            emotion_descriptions = {
                "happily surprised": "Joy mixed with surprise - raised eyebrows and smiling mouth",
                "happily disgusted": "Joy mixed with disgust - smile with nose wrinkle",
                "sadly fearful": "Sadness mixed with fear - furrowed brows and wide eyes",
                "sadly angry": "Sadness mixed with anger - downturned lips and tense face",
                "sadly surprised": "Sadness mixed with surprise - raised brows with sad eyes",
                "sadly disgusted": "Sadness mixed with disgust - sad expression with nose wrinkle",
                "fearfully angry": "Fear mixed with anger - wide eyes with tense expression",
                "fearfully surprised": "Fear mixed with surprise - wide eyes with raised brows",
                "fearfully disgusted": "Fear mixed with disgust - fearful expression with nose wrinkle",
                "angrily surprised": "Anger mixed with surprise - tense brows with raised eyes",
                "angrily disgusted": "Anger mixed with disgust - angry expression with nose wrinkle",
                "disgustedly surprised": "Disgust mixed with surprise - wrinkled nose with raised brows",
                "happily fearful": "Joy mixed with fear - smile with wide eyes",
                "happily sad": "Joy mixed with sadness - smile with sad eyes"
            }
            
            emotion_desc = emotion_descriptions.get(result['emotion'], "Unknown emotion")
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 📝 Description")
            st.markdown(emotion_desc)
            st.markdown('</div>', unsafe_allow_html=True)
        
        progress_bar.progress(80)
        
        # Generate explanation
        if generate_explanation:
            status_text.text("✍️ Generating explanation...")
            
            explanation = model.generate_emotion_explanation(image, result['emotion'])
            
            progress_bar.progress(90)
            
            st.markdown("---")
            st.subheader("📝 Explanation")
            
            st.markdown(f"> {explanation}")
            
            # Evaluate explanation quality
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("📊 Explanation Quality")
            
            # Create dummy reference for evaluation
            reference = f"The person shows {result['emotion']}."
            
            # Calculate metrics
            metrics_calculator = TextualMetrics(use_bleu=True, use_rouge=False)
            bleu_scores = metrics_calculator.compute_bleu([reference], [explanation])
            
            st.metric("BLEU-4", f"{bleu_scores.get('BLEU-4', 0):.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        progress_bar.progress(100)
        status_text.text("✅ Classification complete!")
        
        # Show XAI visualization
        if show_xai:
            st.markdown("---")
            st.subheader("🔍 Visual Interpretation")
            
            # Generate Grad-CAM
            try:
                # Note: This would require the model to have a vision encoder
                st.info("⚠️ XAI visualization requires model with accessible vision encoder.")
                st.info("📌 For full XAI, please use the training scripts with XAI enabled.")
            except Exception as e:
                st.warning(f"⚠️ Could not generate XAI visualization: {e}")
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Classify Another"):
                st.rerun()
        
        with col2:
            if st.button("📥 Download Results"):
                download_results(result, explanation if generate_explanation else None)
        
        with col3:
            if st.button("📊 View All Metrics"):
                show_all_metrics(result)
    
    except Exception as e:
        st.error(f"❌ Error during classification: {e}")
        import traceback
        st.error(traceback.format_exc())


def download_results(result, explanation):
    """Download results as text file."""
    results_text = f"""
Vision-LLM Emotion Recognition Results
{'='*50}

Predicted Emotion: {result['emotion']}
Confidence: {result.get('confidence', 'N/A')}
{'='*50}

Explanation:
{explanation}

{'='*50}
Generated by Vision-LLM Compound Emotion Recognition System
"""
    
    st.download_button(
        label="📥 Download Results",
        data=results_text,
        file_name="emotion_recognition_results.txt",
        mime="text/plain"
    )


def show_all_metrics(result):
    """Show all available metrics."""
    st.markdown("---")
    st.subheader("📊 All Metrics")
    
    metrics = {
        "Predicted Emotion": result.get('emotion', 'N/A'),
        "Confidence": f"{result.get('confidence', 0) * 100:.2f}%",
        "Model": result.get('model', 'Unknown')
    }
    
    for metric, value in metrics.items():
        st.metric(metric, value)


def benchmark_results_page():
    """Benchmark results page."""
    st.header("📊 Benchmark Results")
    
    st.markdown("""
    This page displays benchmark results comparing different models.
    
    ### How to Generate Results
    
    Run the benchmark script to generate results:
    
    ```bash
    python training/benchmark.py --include_vision_only --include_vision_llm --generate_explanations
    ```
    
    Then upload the generated JSON file below to visualize results.
    """)
    
    # Upload benchmark results
    uploaded_file = st.file_uploader(
        "Upload benchmark results JSON",
        type=["json"],
        accept_multiple_files=False,
        help="Upload the benchmark_results_*.json file generated by the benchmark script"
    )
    
    if uploaded_file is not None:
        import json
        try:
            with open(uploaded_file, 'r') as f:
                results = json.load(f)
            
            # Display results
            st.subheader("📊 Model Performance Comparison")
            
            # Create comparison table
            model_names = list(results.keys())
            metrics = ['accuracy', 'f1_macro', 'f1_weighted', 
                      'avg_inference_time_ms', 'throughput_img_per_sec']
            
            # Display metrics
            for metric in metrics:
                st.markdown(f"### {metric.replace('_', ' ').title()}")
                
                data = {model: results[model].get(metric, 0) 
                        for model in model_names}
                
                # Format for display
                if 'time' in metric.lower():
                    formatted_data = {k: f"{v:.1f} ms" for k, v in data.items()}
                elif 'throughput' in metric.lower():
                    formatted_data = {k: f"{v:.1f} img/s" for k, v in data.items()}
                else:
                    formatted_data = {k: f"{v*100:.2f}%" for k, v in data.items()}
                
                st.dataframe(formatted_data, use_container_width=True)
            
            # Visualizations
            st.subheader("📈 Visualizations")
            
            # Accuracy comparison
            fig_col1, fig_col2 = st.columns(2)
            
            with fig_col1:
                st.markdown("#### Accuracy Comparison")
                accuracies = [results[m].get('accuracy', 0) * 100 for m in model_names]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['#e74c3c' if 'Vision' in m else '#2ecc71' for m in model_names]
                bars = ax.bar(model_names, accuracies, color=colors)
                
                ax.set_ylabel('Accuracy (%)')
                ax.set_xlabel('Model')
                ax.set_title('Model Accuracy Comparison')
                ax.set_ylim([0, 100])
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center', va='bottom')
                
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                st.image(buf.getvalue(), caption="Accuracy Comparison", use_column_width=True)
                plt.close()
            
            with fig_col2:
                st.markdown("#### F1 Score Comparison")
                f1_scores = [results[m].get('f1_macro', 0) * 100 for m in model_names]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(model_names, f1_scores, color='#3498db')
                
                ax.set_ylabel('F1 Score (%)')
                ax.set_xlabel('Model')
                ax.set_title('F1 Score Comparison')
                ax.set_ylim([0, 100])
                ax.grid(axis='y', alpha=0.3)
                
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                st.image(buf.getvalue(), caption="F1 Score Comparison", use_column_width=True)
                plt.close()
            
            # Best model summary
            st.subheader("🏆 Best Models by Metric")
            
            best_accuracy = max(results.items(), key=lambda x: x[1].get('accuracy', 0))
            best_f1 = max(results.items(), key=lambda x: x[1].get('f1_macro', 0))
            fastest = min(results.items(), key=lambda x: x[1].get('avg_inference_time_ms', float('inf')))
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Best Accuracy", 
                         f"{best_accuracy[1].get('accuracy', 0)*100:.2f}%",
                         help=f"Model: {best_accuracy[0]}")
            
            with col2:
                st.metric("Best F1 Score", 
                         f"{best_f1[1].get('f1_macro', 0)*100:.2f}%",
                         help=f"Model: {best_f1[0]}")
            
            with col3:
                st.metric("Fastest Model", 
                         f"{fastest[1].get('avg_inference_time_ms', 0):.1f} ms",
                         help=f"Model: {fastest[0]}")
        
        except Exception as e:
            st.error(f"❌ Error loading benchmark results: {e}")


def about_page():
    """About page."""
    st.header("📖 About")
    
    st.markdown("""
    ## Vision-LLM Compound Emotion Recognition
    
    ### Overview
    
    This project implements a Vision-Language Model (Vision-LLM) approach for recognizing 
    **compound emotions** in facial expressions. Unlike traditional Facial Expression Recognition (FER) 
    systems that classify basic emotions (anger, joy, sadness, fear, disgust, surprise), 
    our approach can understand and explain complex emotional states that combine multiple basic emotions.
    
    ### Compound Emotions
    
    The RAF-CE dataset contains 14 compound emotion categories:
    
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
    
    ### Vision-LLM Models
    
    This project supports multiple Vision-LLM architectures:
    
    - **BLIP-2:** Excellent image-text understanding and detailed captioning
    - **Qwen-VL:** Strong performance on micro-expressions with robust reasoning
    - **InternVL:** Powerful vision capabilities with detailed analysis
    - **CLIP:** Zero-shot classification with strong vision-text alignment
    - **Ensemble:** Combines predictions from multiple models for robustness
    
    ### Evaluation Metrics
    
    - **Classification Metrics:** Accuracy, F1-Score (Macro/Weighted), Confusion Matrix
    - **Textual Metrics:** BLEU, ROUGE, CLIPScore, Faithfulness
    - **Efficiency Metrics:** Inference time, throughput, memory usage
    
    ### Citation
    
    If you use this project in your research, please cite:
    
    ```bibtex
    @software{vision_llm_fer,
      title={Vision-LLM for Compound Emotion Recognition},
      author={[Your Name]},
      year={2024},
      note={RAF-CE Dataset},
      url={https://github.com/medmaamar1/emotion-recognition-ai}
    }
    ```
    
    ### Contact
    
    For questions or feedback, please open an issue on GitHub:
    [https://github.com/medmaamar1/emotion-recognition-ai/issues]
    """)


if __name__ == "__main__":
    main()
