import os
import subprocess
import streamlit as st

st.set_page_config(page_title="AI Audio Stem Separator", layout="centered")

st.title("🎵 AI Music Stem Separator")
st.write("Upload an audio file to isolate vocals, drums, bass, and other instruments using Meta's Demucs model.")

uploaded_file = st.file_uploader("Choose an audio file (MP3/WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    # Save uploaded file locally
    input_path = os.path.join("temp_input", uploaded_file.name)
    os.makedirs("temp_input", exist_ok=True)
    
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.audio(input_path)
    
    if st.button("Separate Stems"):
        with st.spinner("Processing audio with AI... This may take 1-2 minutes."):
            output_dir = "separated_stems"
            # Run Demucs CLI command
            cmd = f"demucs -n htdemucs --out {output_dir} {input_path}"
            subprocess.run(cmd, shell=True, check=True)
            
            # Find generated stem files
            filename = os.path.splitext(uploaded_file.name)[0]
            stem_folder = os.path.join(output_dir, "htdemucs", filename)
            
            st.success("Separation Complete!")
            
            # Display individual audio players for each instrument stem
            stems = ["vocals", "drums", "bass", "other"]
            for stem in stems:
                stem_file = os.path.join(stem_folder, f"{stem}.wav")
                if os.path.exists(stem_file):
                    st.subheader(f"🎼 {stem.capitalize()}")
                    st.audio(stem_file)
