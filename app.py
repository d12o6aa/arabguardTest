import streamlit as st
from transformers import pipeline
from PIL import Image

# --- إعدادات الصفحة ---
st.set_page_config(page_title="Arabic/English OCR", page_icon="📝")

# --- تحميل موديل الـ OCR (بدون توكين) ---
@st.cache_resource
def load_ocr_pipeline():

    ocr_pipe = pipeline("image-to-text", model="microsoft/trocr-base-printed")
    return ocr_pipe

with st.spinner('جارٍ تجهيز محرك القراءة...'):
    pipe = load_ocr_pipeline()

# --- واجهة المستخدم ---
st.title("📝 استخراج النصوص من الصور (OCR)")
st.write("ارفعي صورة تحتوي على نص إنجليزي وسأقوم بتحويلها إلى نص مكتوب.")

uploaded_file = st.file_uploader("اختر صورة النص...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="الصورة المرفوعة", use_container_width=True)
    
    if st.button("استخراج النص"):
        with st.spinner('جارٍ القراءة وتحليل الحروف...'):
            # تنفيذ الـ OCR
            result = pipe(image)
            
            st.divider()
            st.subheader("النص المستخرج:")
            
            # عرض النص الناتج
            if result:
                extracted_text = result[0]['generated_text']
                st.success(extracted_text)
                
                # زر لنسخ النص
                st.button("نسخ النص", on_click=lambda: st.write(f"تم النسخ: {extracted_text}"))
            else:
                st.warning("لم أتمكن من التعرف على نص في هذه الصورة.")

else:
    st.info("ارفعي صورة ورقة أو مستند لبدء الاستخراج.")