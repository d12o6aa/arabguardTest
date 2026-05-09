import streamlit as st
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import numpy as np
from datasets import load_dataset

# --- Page Configuration ---
st.set_page_config(page_title="AI Voice Generator", page_icon="🔊")

@st.cache_resource
def load_tts_models():
    # تحميل المكونات يدويًا لضمان استقرار الأداء
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    return processor, model, vocoder

with st.spinner('Loading Voice Engine...'):
    processor, model, vocoder = load_tts_models()

# --- User Interface ---
st.title("🔊 AI Voice Generator (TTS)")

text_input = st.text_area("Enter English Text:", placeholder="Type here...")

if st.button("Generate Voice"):
    if text_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        with st.spinner('Synthesizing...'):
            try:
                # حل مشكلة الـ Dataset: تحميل الـ xvector بطريقة متوافقة مع التحديثات الجديدة
                # هنضيف trust_remote_code=True عشان نتخطى الخطأ
                embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True)
                speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

                # معالجة النص
                inputs = processor(text=text_input, return_tensors="pt")

                # توليد الصوت باستخدام الموديل والـ vocoder
                speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

                # عرض الصوت
                st.audio(speech.numpy(), format="audio/wav", sample_rate=16000)
                st.success("Done!")
            except Exception as e:
                st.error(f"Error during synthesis: {e}")