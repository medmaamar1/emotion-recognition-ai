import streamlit as st
from PIL import Image
import os
import torch
import numpy as np
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

st.set_page_config(page_title="FER-CE: Compound Emotion AI", layout="wide")

st.title("🎭 Facial Emotion Recognition of Compound Expressions")
st.markdown("""
This application demonstrates a Vision-LLM pipeline for recognizing and explaining complex human emotions.
""")

# Sidebar for navigation and model selection
st.sidebar.title("Configuration")
model_type = st.sidebar.selectbox("Select Model", ["ResNet50 (Baseline)", "ViT (Baseline)", "LLaVA-1.5 (Vision-LLM)"])
show_xai = st.sidebar.checkbox("Show XAI (Grad-CAM)", value=True)

# Main columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input Image")
    uploaded_file = st.file_uploader("Choose a facial image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
    else:
        # Sample image
        sample_path = r"c:\Users\OrdiOne\Desktop\emotion recognition ai\RAF-AU\aligned\0001_aligned.jpg"
        if os.path.exists(sample_path):
            st.image(Image.open(sample_path), caption="Sample Aligned Image (0001)", use_column_width=True)

with col2:
    st.header("Model Analysis")
    if uploaded_file is not None or st.button("Analyze Sample"):
        with st.spinner("Analyzing..."):
            # Placeholder for actual model inference
            st.success("Analysis Complete!")
            
            emotion_label = "Happily Surprised"
            confidence = 89.4
            
            st.subheader(f"Prediction: {emotion_label}")
            st.progress(confidence / 100)
            
            st.write("---")
            st.subheader("Explanation")
            st.write("**Action Units Activated:** AU1 (Inner Brow Raiser), AU2 (Outer Brow Raiser), AU6 (Cheek Raiser), AU12 (Lip Corner Puller).")
            st.write("**Interpretation:** The upward curve of the lips combined with the widened eyes indicates a mixture of joy and surprise.")
            
            if show_xai:
                st.write("---")
                st.subheader("Visual Explanation (Grad-CAM)")
                st.info("Heatmap visualization would appear here in the full implementation.")

st.sidebar.write("---")
st.sidebar.write("**Dataset**: RAF-CE (14 Compound Emotions)")
st.sidebar.write("**Status**: Training in progress...")
