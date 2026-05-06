import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

st.set_page_config(page_title="ArabGuard Detector", page_icon="🛡️")


HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxx"

@st.cache_resource
def load_model(token):
    model_id = "d12o6aa/ArabGuard"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, token=token)
    return tokenizer, model

try:
    tokenizer, model = load_model(HF_TOKEN)
except Exception as e:
    st.error(f"فشل في الوصول للموديل: {e}")
    st.stop()

st.title("🛡️ ArabGuard Content Moderator")
st.write("فحص النصوص باستخدام Access Token للوصول للموديل.")

user_input = st.text_area("النص المراد فحصه:", placeholder="اكتب رسالتك هنا...")

if st.button("تحليل النص"):
    if user_input.strip() == "":
        st.warning("من فضلك اكتب نصاً أولاً!")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=64)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            prediction = torch.argmax(logits, dim=-1).item()

        st.divider()
        if prediction == 1:
            st.error("⚠️ **النتيجة: Blocked (Malicious)**")
        else:
            st.success("✅ **النتيجة: Safe**")

st.caption("Access secured via Hugging Face Token")