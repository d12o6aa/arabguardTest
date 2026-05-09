import streamlit as st
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import re
import gc # عشان ننظف الرامات أول بأول

st.set_page_config(page_title="Lite AI OCR", page_icon="📝")

@st.cache_resource
def load_ocr_model():
    model_id = "naver-clova-ix/donut-base-finetuned-docvqa"
    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

processor, model, device = load_ocr_model()

st.title("📝 Optimized AI OCR")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # --- خطوة مهمة جداً: تصغير حجم الصورة لتقليل استهلاك الرامات ---
    max_size = 800
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    st.image(image, caption="Resized for stability", use_container_width=True)
    
    if st.button("Extract Text"):
        with st.spinner('Processing...'):
            try:
                # تحضير الداتا
                task_prompt = "<s_docvqa>"
                decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
                pixel_values = processor(image, return_tensors="pt").pixel_values

                # الإعدادات دي بتخلي الاستهلاك أقل
                outputs = model.generate(
                    pixel_values.to(device),
                    decoder_input_ids=decoder_input_ids.to(device),
                    max_length=256, # قللت الـ length عشان الرامات
                    use_cache=True,
                    return_dict_in_generate=True,
                )

                sequence = processor.batch_decode(outputs.sequences)[0]
                extracted_text = re.sub(r"<.*?>", "", sequence).strip()

                st.success(f"Result: {extracted_text}")
                
                # تنظيف الذاكرة يدوياً بعد كل عملية
                del pixel_values, outputs, sequence
                gc.collect() 

            except Exception as e:
                st.error("Memory limit reached or error occurred. Try a smaller image.")