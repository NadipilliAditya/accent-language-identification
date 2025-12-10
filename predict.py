import torch
import librosa
import numpy as np
import sounddevice as sd
from sklearn.preprocessing import StandardScaler

# Load Model + Scaler
model = torch.load("accent_model_svm.pkl")
scaler = torch.load("feature_scaler.pkl")

def extract_mfcc_from_audio(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
    return np.mean(mfcc, axis=1)

def record_audio(duration=3, sr=16000):
    print("\n🎙️ Speak now... (Recording for 3 seconds)\n")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    return audio.flatten()

audio = record_audio()
features = extract_mfcc_from_audio(audio)
features = scaler.transform([features])
prediction = model.predict(features)[0]

print("\n🔊 **Predicted Accent:**", prediction.upper())
