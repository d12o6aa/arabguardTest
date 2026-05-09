import streamlit as st
from transformers import pipeline
from PIL import Image

# --- Page Configuration ---
st.set_page_config(page_title="AI Text Extractor", page_icon="📝")

# --- Load OCR Pipeline (Optimized Small Model) ---
@st.cache_resource
def load_ocr_pipeline():
    # Using the 'small' version for faster performance
    ocr_pipe = pipeline("image-to-text", model="microsoft/trocr-small-printed")
    return ocr_pipe

with st.spinner('Loading OCR Engine...'):
    pipe = load_ocr_pipeline()

# --- User Interface ---
st.title("📝 Image to Text Extractor (OCR)")
st.write("Upload an image containing English text to convert it into editable text.")

uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Extract Text"):
        with st.spinner('Analyzing characters...'):
            # Run the OCR inference
            result = pipe(image)
            
            st.divider()
            st.subheader("Extracted Text:")
            
            # Check and display the result
            if result and result[0]['generated_text'].strip():
                extracted_text = result[0]['generated_text']
                st.success(extracted_text)
                
                # Simple display for copying
                st.info("You can highlight and copy the text above.")
            else:
                st.warning("No clear text was detected in this image.")

else:
    st.info("Please upload a document or image to start the extraction.")