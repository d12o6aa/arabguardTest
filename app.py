import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# --- Page Configuration ---
st.set_page_config(page_title="AI Text Extractor", page_icon="📝")

# --- Load Model and Processor Separately (More Stable) ---
@st.cache_resource
def load_ocr_model():
    model_id = "microsoft/trocr-small-printed"
    # Loading processor and model manually to avoid 'KeyError' in pipeline
    processor = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    return processor, model

with st.spinner('Loading OCR Engine...'):
    processor, model = load_ocr_model()

# --- User Interface ---
st.title("📝 Image to Text Extractor (OCR)")
st.write("Upload an image containing English text to convert it into editable text.")

uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Extract Text"):
        with st.spinner('Analyzing characters...'):
            # Prepare the image for the model
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            
            # Generate text
            generated_ids = model.generate(pixel_values)
            extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            st.divider()
            st.subheader("Extracted Text:")
            
            if extracted_text.strip():
                st.success(extracted_text)
                st.info("You can highlight and copy the text above.")
            else:
                st.warning("No clear text was detected in this image.")
else:
    st.info("Please upload a document or image to start the extraction.")