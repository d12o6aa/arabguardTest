import streamlit as st
import easyocr
from PIL import Image
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="AI Text Extractor (Fix)", page_icon="📝")

# --- Load EasyOCR Reader (Caching for performance) ---
@st.cache_resource
def load_ocr_reader():
    # 'en' specifies English. You can add other languages like ['en', 'ar'].
    # Using 'gpu=False' is safer for general compatibility, 
    # but set it to True if you have a CUDA GPU.
    reader = easyocr.Reader(['en'], gpu=False) 
    return reader

with st.spinner('Loading OCR Engine...'):
    reader = load_ocr_reader()

# --- User Interface ---
st.title("📝 Dedicated Image-to-Text Extractor")
st.write("Upload any complex screenshot to extract all English text accurately.")

uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Extract Text"):
        with st.spinner('Analyzing... (this might take a moment)'):
            # EasyOCR requires a numpy array or a file path.
            # Convert PIL image to NumPy array.
            image_np = np.array(image_pil)
            
            # Read text
            # result is a list of tuples: (bounding_box, text, confidence_score)
            results = reader.readtext(image_np)
            
            st.divider()
            st.subheader("Extracted Text:")
            
            if results:
                # Join all detected text segments into a single string
                extracted_text = " ".join([res[1] for res in results])
                
                st.success("Extraction Complete!")
                # Use st.text_area for easy copying of long text (like code)
                st.text_area("Result:", extracted_text, height=300)
            else:
                st.warning("No clear text was detected in this image.")

else:
    st.info("Please upload a document or image to start the extraction.")