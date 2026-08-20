import os
import sys
import torch
import streamlit as st
from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.audio import AudioFile, save_audio

# Page Configuration
st.set_page_config(page_title="AI Instrument Separator", layout="centered")

# Custom Clean Solid Light-Blue & Darker Blue Accent Styling (CSS)
st.markdown(
    """
    <style>
    /* Clean solid light-blue background */
    .stApp {
        background-color: #e2ebf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Darker Blue Primary Buttons */
    div.stButton > button {
        background-color: #004080 !important;
        color: white !important;
        border-radius: 6px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
        transition: background-color 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: #002b55 !important;
        color: white !important;
    }

    /* Container card styling */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.8);
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("AI Instrument Separator")
st.write("Isolate individual track components (Drums, Bass, Guitar, Vocals, Piano, and Other) from audio files using deep learning.")

# File Upload Section
uploaded_file = st.file_uploader("Choose an audio file (MP3/WAV)", type=["mp3", "wav"])

input_path = None
temp_dir = "temp_input"
os.makedirs(temp_dir, exist_ok=True)

if uploaded_file is not None:
    input_path = os.path.join(temp_dir, uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(input_path)

# Run AI Separation
if input_path and os.path.exists(input_path):
    if st.button("Separate Instruments"):
        with st.spinner("Processing audio with 6-stem AI model... This may take 1-2 minutes."):
            try:
                # Load Demucs 6-stem neural network (htdemucs_6s)
                model = get_model('htdemucs_6s')
                model.cpu()
                
                # Load audio data
                wav = AudioFile(input_path).read(
                    streams=0, 
                    samplerate=model.samplerate, 
                    channels=model.audio_channels
                )
                
                # Preprocess waveform
                ref = wav.mean(0)
                wav = (wav - ref.mean()) / ref.std()
                
                # Perform AI inference
                with torch.no_grad():
                    sources = apply_model(model, wav[None], shifts=0, split=True, overlap=0.25)[0]
                
                sources = sources * ref.std() + ref.mean()
                
                # Export separated audio files
                output_dir = "separated_instruments"
                os.makedirs(output_dir, exist_ok=True)
                
                # Native output mapping from htdemucs_6s:
                # 0: Drums, 1: Bass, 2: Other, 3: Vocals, 4: Guitar, 5: Piano
                native_order = ["Drums", "Bass", "Other", "Vocals", "Guitar", "Piano"]
                mapped_sources = dict(zip(native_order, sources))
                
                # Desired display order
                display_order = ["Drums", "Bass", "Guitar", "Vocals", "Piano", "Other"]
                
                st.success("Instrument separation complete.")
                
                # Display output audio players in the requested sequence
                for name in display_order:
                    source = mapped_sources[name]
                    file_path = os.path.join(output_dir, f"{name.lower()}.wav")
                    save_audio(source.cpu(), file_path, samplerate=model.samplerate)
                    
                    st.markdown(f"### {name}")
                    st.audio(file_path)

            except Exception as e:
                st.error(f"Error executing instrument separation: {e}")
