import streamlit as st
import librosa
import numpy as np
import joblib

# Load Model + Scaler
model = joblib.load("accent_model_svm.pkl")
scaler = joblib.load("feature_scaler.pkl")

SAMPLE_RATE = 16000

def extract_mfcc(path):
    audio, sr = librosa.load(path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc, axis=1)

st.title("🎙️ Indian Accent Recognition - File Upload Mode")
st.write("Upload a `.wav` voice clip and I will predict the accent 🤖")

uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])

if uploaded_file is not None:
    st.audio(uploaded_file)

    try:
        features = extract_mfcc(uploaded_file)
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]

        st.success(f"🔊 **Predicted Accent:** {prediction.upper()}")

    except Exception as e:
        st.error("⚠️ Could not process the audio. Make sure it is a clear speech WAV file.")
        st.error(str(e))

st.markdown("---")
st.write("Made with ❤️ by **Aditya**")
