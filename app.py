import streamlit as st
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="AI Voice Generator", page_icon="🔊")

@st.cache_resource
def load_tts_models():
    # تحميل الموديلات بشكل منفصل
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    return processor, model, vocoder

processor, model, vocoder = load_tts_models()

st.title("🔊 AI Voice Generator (TTS)")
text_input = st.text_area("Enter English Text:", placeholder="Type here...")

if st.button("Generate Voice"):
    if text_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        with st.spinner('Synthesizing...'):
            try:
                # بدل تحميل الـ dataset، هنستخدم بصمة صوت ثابتة (Default Speaker)
                # دي عينة من الـ xvector الخاص بموديل SpeechT5
                speaker_embeddings = torch.zeros((1, 512)) # بصمة صوت محايدة
                
                # معالجة النص
                inputs = processor(text=text_input, return_tensors="pt")

                # توليد الصوت باستخدام الـ vocoder
                speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

                # عرض النتيجة
                st.audio(speech.numpy(), format="audio/wav", sample_rate=16000)
                st.success("Voice generated successfully!")
                
            except Exception as e:
                st.error(f"Error during synthesis: {e}")