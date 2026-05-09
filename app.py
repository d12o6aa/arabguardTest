import streamlit as st
from transformers import pipeline
import torch
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="AI Voice Generator", page_icon="🔊")

# --- Load TTS Pipeline ---
@st.cache_resource
def load_tts_pipeline():
    # نستخدم SpeechT5 للتحويل من نص لحديث (Text-to-Speech)
    tts_pipe = pipeline("text-to-speech", model="microsoft/speecht5_tts")
    return tts_pipe

with st.spinner('Loading Voice Engine...'):
    synthesizer = load_tts_pipeline()

# --- User Interface ---
st.title("🔊 AI Voice Generator (TTS)")
st.write("Enter English text below to generate a natural-sounding voice.")

# Input text
text_input = st.text_area("Enter Text:", placeholder="Type something here, e.g., 'Hello Doaa, how is your ArabGuard project going?'")

if st.button("Generate Voice"):
    if text_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner('Synthesizing speech...'):
            # تحميل الـ speaker embeddings (ضروري لموديل SpeechT5 عشان يحدد نبرة الصوت)
            # هنستخدم ملف صوت افتراضي من Hugging Face لضبط النبرة
            from datasets import load_dataset
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

            # توليد الصوت
            speech = synthesizer(text_input, forward_params={"speaker_embeddings": speaker_embeddings})
            
            # استخراج البيانات الصوتية ومعدل العينة (Sampling Rate)
            audio_data = speech["audio"]
            sampling_rate = speech["sampling_rate"]

            # عرض النتائج
            st.divider()
            st.success("Voice generated successfully!")
            
            # تشغيل الصوت في المتصفح
            st.audio(audio_data, format="audio/wav", sample_rate=sampling_rate)

else:
    st.info("Ready to turn your text into speech.")