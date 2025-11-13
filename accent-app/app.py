import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import joblib
import pandas as pd
import datetime
import os
import sounddevice as sd
import tempfile

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Accent / Language Identification",
    page_icon="🎧",
    layout="centered",
)

# -----------------------------
# Lavender Glow Background & UI Styling
# -----------------------------
page_bg = """
<style>
/* --- Glowing Background --- */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 25% 25%, #f7f3ff 0%, #e4d7ff 40%, #cdb9ff 80%, #bda5ff 100%);
    background-attachment: fixed;
    color: #000000;
    position: relative;
}

/* --- Ambient Glow Layers --- */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: -50px;
    left: -50px;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(186,140,255,0.45) 0%, transparent 70%);
    filter: blur(90px);
    z-index: 0;
}

[data-testid="stAppViewContainer"]::after {
    content: "";
    position: absolute;
    bottom: -100px;
    right: -100px;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(210,160,255,0.4) 0%, transparent 75%);
    filter: blur(100px);
    z-index: 0;
}

/* --- Header --- */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

h1, h2, h3 {
    color: #2e1a47;
    text-align: center;
    font-family: 'Poppins', sans-serif;
    position: relative;
    z-index: 1;
}

/* --- Buttons --- */
.stButton>button {
    background: linear-gradient(135deg, #7c3aed, #8b5cf6, #a78bfa);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 28px;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4);
    position: relative;
    z-index: 1;
}
.stButton>button:hover {
    transform: scale(1.07);
    box-shadow: 0 6px 25px rgba(139,92,246,0.55);
}

/* --- Cards (Glass + Glow) --- */
.result-card {
    background: rgba(255, 255, 255, 0.35);
    backdrop-filter: blur(14px);
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 8px 25px rgba(139,92,246,0.3);
    border: 1px solid rgba(255,255,255,0.35);
    margin-top: 25px;
    text-align: center;
    transition: 0.4s;
}
.result-card:hover {
    transform: scale(1.03);
    box-shadow: 0 12px 35px rgba(139,92,246,0.45);
}

/* --- Alerts / Messages --- */
div.stAlert {
    border-radius: 12px;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.08);
    z-index: 1;
}

/* --- Tabs Glow --- */
[data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.4);
    border-radius: 10px;
    backdrop-filter: blur(10px);
    padding: 4px;
    box-shadow: 0 2px 10px rgba(139,92,246,0.25);
}

hr {
    border: 1px solid #d8cfff;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -----------------------------
# Load trained model and tools
# -----------------------------
try:
    model = joblib.load("rf_mfcc_model.joblib")
    scaler = joblib.load("scaler.joblib")
    encoder = joblib.load("label_encoder.joblib")
    st.success("✅ Model and encoders loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# -----------------------------
# Utility Functions
# -----------------------------
SAMPLE_RATE = 16000
RECORD_DURATION = 3  # seconds

def save_bytes_to_temp_wav(wav_bytes: bytes) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(wav_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name

def save_numpy_audio_to_temp_wav(np_audio: np.ndarray, sr: int) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, np_audio, sr)
    tmp.close()
    return tmp.name

def extract_mfcc_from_file(path: str, n_mfcc: int = 13):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean, sr

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎧 Accent / Language Identification App")
st.write("Upload a `.wav` file or record your voice below to detect the accent or language.")

tab1, tab2 = st.tabs(["📁 Upload Audio", "🎤 Record Voice"])

audio_file_path = None
uploaded_file = None

# -----------------------------
# OPTION 1: Upload Audio File
# -----------------------------
with tab1:
    uploaded_file = st.file_uploader("Choose a `.wav` file", type=["wav"])
    if uploaded_file is not None:
        try:
            uploaded_bytes = uploaded_file.read()
            audio_file_path = save_bytes_to_temp_wav(uploaded_bytes)
            st.audio(uploaded_bytes, format="audio/wav")
            st.success("✅ Uploaded audio ready.")
        except Exception as e:
            st.error(f"❌ Failed to process uploaded file: {e}")
            audio_file_path = None

# -----------------------------
# OPTION 2: Record Audio (sounddevice)
# -----------------------------
with tab2:
    st.write(f"Click below and speak for {RECORD_DURATION} seconds 🎙️")
    if st.button("🎙️ Start Recording"):
        try:
            st.info("Recording... Speak now!")
            recorded = sd.rec(int(RECORD_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            sd.wait()
            st.success("✅ Recording complete!")
            recorded_flat = recorded.flatten()
            audio_file_path = save_numpy_audio_to_temp_wav(recorded, SAMPLE_RATE)
            with open(audio_file_path, "rb") as f:
                st.audio(f.read(), format="audio/wav")
        except Exception as e:
            st.error(f"❌ Recording failed: {e}")
            audio_file_path = None

# -----------------------------
# PROCESS AUDIO IF AVAILABLE
# -----------------------------
if audio_file_path is not None:
    try:
        mfcc_mean, sr = extract_mfcc_from_file(audio_file_path, n_mfcc=13)
        audio_len_seconds = sf.info(audio_file_path).duration

        if audio_len_seconds < 0.5:
            st.warning("⚠️ Audio too short! Please record at least 1 second.")
        else:
            mfcc_scaled = mfcc_mean.reshape(1, -1)
            features_scaled = scaler.transform(mfcc_scaled)
            prediction = model.predict(features_scaled)
            predicted_label = encoder.inverse_transform(prediction)[0]
            confidence = (
                float(np.max(model.predict_proba(features_scaled))) * 100
                if hasattr(model, "predict_proba")
                else 0.0
            )

            # Logging
            log_data = {
                "timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                "source": ["Uploaded" if uploaded_file else "Recorded"],
                "predicted_language": [predicted_label],
                "confidence": [f"{confidence:.2f}"]
            }
            log_df = pd.DataFrame(log_data)
            if not os.path.exists("user_predictions.csv"):
                log_df.to_csv("user_predictions.csv", index=False)
            else:
                log_df.to_csv("user_predictions.csv", mode="a", header=False, index=False)

            # --- Glowing Result Card ---
            st.markdown(
                f"""
                <div class="result-card">
                    <h3 style="color:#4c1d95;">🎯 Predicted Accent/Language</h3>
                    <h2 style="color:#6d28d9;">{predicted_label}</h2>
                    <p style="font-size:18px;">Confidence: <b>{confidence:.2f}%</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # --- Cuisine Recommendation ---
            st.markdown("### 🍴 Accent-Aware Cuisine Recommendation")
            cuisine_dict = {
                "English": ["Grilled Chicken", "Tacos", "Pizza", "Burger"],
                "Tamil": ["Dosa", "Idli", "Sambar", "Rasam"],
                "Hindi": ["Butter Chicken", "Paneer Tikka", "Dal Makhani"],
                "Telugu": ["Pesarattu", "Pulihora", "Gutti Vankaya Curry"],
                "Kannada": ["Ragi Mudde", "Bisi Bele Bath", "Neer Dosa"],
                "Malayalam": ["Appam", "Puttu", "Avial"],
                "French": ["Croissant", "Ratatouille", "Crème Brûlée"],
                "Spanish": ["Paella", "Tapas", "Churros"],
                "Japanese": ["Sushi", "Ramen", "Tempura"],
                "Chinese": ["Dim Sum", "Fried Rice", "Kung Pao Chicken"],
                "Gujarati": ["Dhokla", "Thepla", "Undhiyu"],
                "Marathi": ["Pav Bhaji", "Misal Pav", "Puran Poli"],
                "Punjabi": ["Sarson da Saag", "Makki di Roti", "Lassi"],
                "Bengali": ["Machher Jhol", "Rasgulla", "Mishti Doi"],
            }

            if predicted_label in cuisine_dict:
                dishes = ", ".join(cuisine_dict[predicted_label])
                st.success(
                    f"Since the detected accent/language is **{predicted_label}**, "
                    f"you might enjoy trying: **{dishes}** 😋"
                )
            else:
                st.info("Cuisine suggestion not available for this accent/language yet.")

            # --- Download Result ---
            result_df = pd.DataFrame({
                "Source": ["Uploaded" if uploaded_file else "Recorded"],
                "Predicted Accent/Language": [predicted_label],
                "Confidence (%)": [f"{confidence:.2f}"]
            })
            st.download_button(
                label="📄 Download Result (CSV)",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="accent_prediction.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"⚠️ Error during processing/prediction: {e}")

st.caption("Note: temporary audio files are stored locally and reused for playback.")
