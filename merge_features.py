import os
import numpy as np
import librosa
from tqdm import tqdm

AUDIO_DIR = "data"   # ✅ correct folder
FEATURE_DIR = "features"
os.makedirs(FEATURE_DIR, exist_ok=True)

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

print("\n🎧 Extracting MFCC features...\n")

for state in sorted(os.listdir(AUDIO_DIR)):
    state_path = os.path.join(AUDIO_DIR, state)
    if not os.path.isdir(state_path):
        continue

    save_folder = os.path.join(FEATURE_DIR, state)
    os.makedirs(save_folder, exist_ok=True)

    count = 0
    for file in tqdm(os.listdir(state_path), desc=f"Processing {state}"):
        if file.lower().endswith(".wav"):
            fp = os.path.join(state_path, file)
            try:
                features = extract_mfcc(fp)
                np.save(os.path.join(save_folder, file.replace(".wav", ".npy")), features)
                count += 1
            except:
                pass

    print(f"✅ {state}: {count} samples processed\n")

print("🎉 MFCC extraction completed successfully!")
