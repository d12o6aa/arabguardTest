import streamlit as st
from transformers import pipeline
from PIL import Image
import torch
import gc

# --- Page Configuration ---
st.set_page_config(page_title="Stable AI OCR", page_icon="🛡️")

# --- Load Model (Small & Stable) ---
@st.cache_resource
def load_stable_model():
    # الموديل ده خفيف جداً ومناسب للـ Cloud RAM
    return pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

# محاولة تحميل الموديل
try:
    pipe = load_stable_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

st.title("🛡️ AI Text Extractor")
st.write("Optimized for Python 3.14 and Streamlit 2026 Standards.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # تصغير الصورة لضمان استقرار الرامات
    image.thumbnail((600, 600))
    
    # التحديث الجديد لـ Streamlit: استخدام width='stretch' بدل use_container_width
    st.image(image, caption="Processed Image", width='stretch')
    
    if st.button("Extract Text"):
        with st.spinner('Analyzing...'):
            try:
                result = pipe(image)
                st.success("Analysis Complete!")
                st.write(f"**Result:** {result[0]['generated_text']}")
                
                # تنظيف الذاكرة
                gc.collect()
            except Exception as e:
                st.error(f"An error occurred during extraction: {e}")