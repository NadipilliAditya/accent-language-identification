import os
import numpy as np
import librosa
from tqdm import tqdm
import torch

# ✅ Correct audio folder
AUDIO_DIR = "IndicAccentDb_full"


# ✅ Features output folder
SAVE_DIR = "features"
os.makedirs(SAVE_DIR, exist_ok=True)

print("\n🎧 Extracting MFCC features...\n")

def extract_mfcc(path):
    y, sr = librosa.load(path, sr=16000)  # Normalize sampling rate
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc, axis=1)  # Average over time → fixed length

for state in sorted(os.listdir(AUDIO_DIR)):
    state_path = os.path.join(AUDIO_DIR, state)

    # Skip if not folder
    if not os.path.isdir(state_path):
        continue

    save_file = os.path.join(SAVE_DIR, f"{state}.pt")

    # ✅ Skip already extracted
    if os.path.exists(save_file):
        print(f"⏩ Skipping {state} (already processed)")
        continue

    print(f"📦 Processing State -> {state}")
    features_list = []

    for file in tqdm(os.listdir(state_path), desc=f"Processing {state}"):
        if file.lower().endswith(".wav"):
            wav_path = os.path.join(state_path, file)
            try:
                feat = extract_mfcc(wav_path)
                features_list.append(feat)
            except:
                print(f"⚠️ Skipped: {file}")

    torch.save(features_list, save_file)
    print(f"✅ Saved features to: {save_file}\n")

print("🎉 Feature extraction complete! Check /features folder.\n")

