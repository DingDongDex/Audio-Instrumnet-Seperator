import os
import sys
import torch
import streamlit as st
from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.audio import AudioFile, save_audio

st.set_page_config(page_title="AI Audio Stem Separator", layout="centered")

st.title("🎵 AI Music Stem Separator")
st.write("Upload an audio file to isolate vocals, drums, bass, and other instruments using Meta's Demucs model.")

uploaded_file = st.file_uploader("Choose an audio file (MP3/WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    # 1. Save uploaded file locally
    temp_dir = "temp_input"
    os.makedirs(temp_dir, exist_ok=True)
    input_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.audio(input_path)
    
    if st.button("Separate Stems"):
        with st.spinner("Processing audio with AI... This may take 1-2 minutes."):
            try:
                # 2. Load lightweight pre-trained Demucs model
                model = get_model('htdemucs')
                model.cpu()  # Force CPU execution for cloud hosting
                
                # 3. Read and process audio
                wav = AudioFile(input_path).read(
                    streams=0, 
                    samplerate=model.samplerate, 
                    channels=model.audio_channels
                )
                
                # Apply model to separate audio into stems
                ref = wav.mean(0)
                wav = (wav - ref.mean()) / ref.std()
                
                with torch.no_grad():
                    sources = apply_model(model, wav[None], shifts=0, split=True, overlap=0.25)[0]
                
                sources = sources * ref.std() + ref.mean()
                
                # 4. Save separated stems to disk
                output_dir = "separated_stems"
                os.makedirs(output_dir, exist_ok=True)
                
                stem_names = ["drums", "bass", "other", "vocals"]
                st.success("Separation Complete!")
                
                # Display individual audio players for each instrument
                for source, name in zip(sources, stem_names):
                    stem_path = os.path.join(output_dir, f"{name}.wav")
                    save_audio(source.cpu(), stem_path, samplerate=model.samplerate)
                    
                    st.subheader(f"🎼 {name.capitalize()}")
                    st.audio(stem_path)

            except Exception as e:
                st.error(f"Error processing audio: {e}")
