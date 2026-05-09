import streamlit as st
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import re

# --- Page Configuration ---
st.set_page_config(page_title="AI Text Extractor", page_icon="📝")

@st.cache_resource
def load_ocr_model():
    # Switching to Donut: It's stable, fast, and great for OCR
    model_id = "naver-clova-ix/donut-base-finetuned-docvqa"
    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

with st.spinner('Loading OCR Engine...'):
    processor, model, device = load_ocr_model()

st.title("📝 Image to Text Extractor (OCR)")
st.write("Upload an image to extract text using the Donut Transformer model.")

uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Extract Text"):
        with st.spinner('Processing document...'):
            # Prepare decoder inputs
            task_prompt = "<s_docvqa>"
            decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
            
            pixel_values = processor(image, return_tensors="pt").pixel_values

            outputs = model.generate(
                pixel_values.to(device),
                decoder_input_ids=decoder_input_ids.to(device),
                max_length=model.config.decoder.max_position_embeddings,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

            sequence = processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
            extracted_text = re.sub(r"<.*?>", "", sequence, count=1).strip() # Clean tags

            st.divider()
            st.subheader("Extracted Result:")
            st.success(extracted_text if extracted_text else "No text detected.")