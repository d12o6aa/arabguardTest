import streamlit as st
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import numpy as np
from datasets import load_dataset

# --- Page Configuration ---
st.set_page_config(page_title="AI Voice Generator", page_icon="🔊")

@st.cache_resource
def load_tts_models():
    # تحميل المكونات يدويًا لضمان استقرار الأداء على السيرفر
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
                # التعديل هنا: إضافة trust_remote_code=True
                embeddings_dataset = load_dataset(
                    "Matthijs/cmu-arctic-xvectors", 
                    split="validation", 
                    trust_remote_code=True
                )
                speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

                # معالجة النص وتحويله لـ Tensors
                inputs = processor(text=text_input, return_tensors="pt")

                # توليد الصوت
                speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

                # عرض الصوت في الـ UI
                st.audio(speech.numpy(), format="audio/wav", sample_rate=16000)
                st.success("Voice generated successfully!")
                
            except Exception as e:
                st.error(f"Error during synthesis: {e}")