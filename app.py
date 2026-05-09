import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# --- Page Configuration ---
st.set_page_config(page_title="Professional OCR", page_icon="📝")

@st.cache_resource
def load_ocr_model():
    # الموديل ده أدق بكتير في التعرف على النصوص المطبوعة
    model_id = "microsoft/trocr-base-printed" 
    processor = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    return processor, model

processor, model = load_ocr_model()

st.title("Professional Image-to-Text Deployment")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Extract Text"):
        with st.spinner('Extracting...'):
            # تحضير الصورة
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            
            # تحديد بارامترات التوليد لضمان عدم الهلوسة
            generated_ids = model.generate(pixel_values, max_new_tokens=100)
            extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            st.success("Extraction Complete!")
            st.markdown(f"### Result:\n **{extracted_text}**")