import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
import gc

# --- Page Configuration ---
st.set_page_config(page_title="Stable AI OCR", page_icon="🛡️")

# --- Load Model Components Separately ---
@st.cache_resource
def load_custom_model():
    model_id = "nlpconnect/vit-gpt2-image-captioning"
    
    # تحميل كل جزء لوحده لضمان التوافق
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    feature_extractor = ViTImageProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, feature_extractor, tokenizer, device

# تحميل المكونات
model, feature_extractor, tokenizer, device = None, None, None, None

try:
    with st.spinner('Loading Model Components...'):
        model, feature_extractor, tokenizer, device = load_custom_model()
except Exception as e:
    st.error(f"Error loading components: {e}")

st.title("🛡️ AI Text Extractor")
st.write("Professional Image-to-Text Deployment (2026 Standards)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image.thumbnail((600, 600))
    st.image(image, caption="Ready for analysis", width='stretch')
    
    if st.button("Extract Text"):
        with st.spinner('Processing...'):
            try:
                # تحويل الصورة لأرقام يفهمها الموديل
                pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(device)

                # توليد النص (Inference)
                output_ids = model.generate(pixel_values, max_length=16, num_beams=4)

                # تحويل الأرقام لكلمات
                preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                result = preds[0].strip()

                st.success("Extraction Complete!")
                st.subheader(f"Result: {result}")
                
                # تنظيف الذاكرة
                gc.collect()
            except Exception as e:
                st.error(f"Inference failed: {e}")