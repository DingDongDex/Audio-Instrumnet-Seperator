import os
import sys
import torch
import streamlit as st
import yt_dlp
from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.audio import AudioFile, save_audio

# Page Configuration
st.set_page_config(page_title="AI Instrument Separator", layout="centered")

# Custom Light Blue Sleek Styling (CSS)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f4f8;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    div.stButton > button {
        background-color: #0066cc;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 8px 16px;
        font-weight: 500;
    }
    div.stButton > button:hover {
        background-color: #0052a3;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("AI Instrument Separator")
st.write("Isolate individual track components (Vocals, Drums, Bass, and Other) from audio files or YouTube links using deep learning.")

# Input Method Selection
input_option = st.radio("Select Input Source:", ["Upload Audio File", "YouTube URL"])

input_path = None
temp_dir = "temp_input"
os.makedirs(temp_dir, exist_ok=True)

if input_option == "Upload Audio File":
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])
    if uploaded_file is not None:
        input_path = os.path.join(temp_dir, uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.audio(input_path)

else:
    youtube_url = st.text_input("Paste YouTube Link:")
    if youtube_url:
        if st.button("Fetch YouTube Audio"):
            with st.spinner("Extracting audio from YouTube..."):
                try:
                    ydl_opts = {
                        'format': 'bestaudio/best',
                        'outtmpl': os.path.join(temp_dir, 'yt_audio.%(ext)s'),
                        'postprocessors': [{
                            'key': 'FFmpegExtractAudio',
                            'preferredcodec': 'mp3',
                            'preferredquality': '192',
                        }],
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([youtube_url])
                    input_path = os.path.join(temp_dir, "yt_audio.mp3")
                    st.success("YouTube audio extracted successfully.")
                    st.audio(input_path)
                except Exception as e:
                    st.error(f"Failed to process YouTube URL: {e}")

# Run AI Separation
if input_path and os.path.exists(input_path):
    if st.button("Separate Instruments"):
        with st.spinner("Processing audio with AI models... This may take 1-2 minutes."):
            try:
                # Load Demucs neural network
                model = get_model('htdemucs')
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
                
                instrument_names = ["Drums", "Bass", "Other", "Vocals"]
                st.success("Instrument separation complete.")
                
                # Display output players
                for source, name in zip(sources, instrument_names):
                    file_path = os.path.join(output_dir, f"{name.lower()}.wav")
                    save_audio(source.cpu(), file_path, samplerate=model.samplerate)
                    
                    st.markdown(f"### {name}")
                    st.audio(file_path)

            except Exception as e:
                st.error(f"Error executing instrument separation: {e}")
