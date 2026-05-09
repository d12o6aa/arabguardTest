import streamlit as st
from transformers import pipeline
from PIL import Image
import torch
import gc

# --- Page Configuration ---
st.set_page_config(page_title="Stable AI OCR", page_icon="🛡️")

# --- Load Model Function ---
@st.cache_resource
def load_stable_model():
    # استخدمنا الاسم الجديد للمهمة المتوافق مع تحديثات 2026
    return pipeline("image-text-to-text", model="nlpconnect/vit-gpt2-image-captioning")

# تعريف المتغير في الـ Global Scope عشان الـ 'Extract' button يشوفه
pipe = None

try:
    with st.spinner('Loading Model...'):
        pipe = load_stable_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

st.title("🛡️ AI Text Extractor")
st.write("Optimized for Stability and Cloud Performance.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image.thumbnail((600, 600))
    st.image(image, caption="Processed Image", width='stretch')
    
    if st.button("Extract Text"):
        # التأكد إن الـ pipe جاهز قبل الاستخدام
        if pipe is not None:
            with st.spinner('Analyzing...'):
                try:
                    result = pipe(image)
                    st.success("Analysis Complete!")
                    # عرض النتيجة
                    st.write(f"**Result:** {result[0]['generated_text']}")
                    gc.collect()
                except Exception as e:
                    st.error(f"Extraction failed: {e}")
        else:
            st.error("Model is not initialized. Please check the logs.")