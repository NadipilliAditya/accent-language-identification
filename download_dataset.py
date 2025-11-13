import os
import requests
import zipfile
from huggingface_hub import hf_hub_download

SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)

print("⬇️ Downloading IndicAccentDb ZIP directly...")

# Download ZIP directly from HF storage
zip_path = hf_hub_download(
    repo_id="DarshanaS/IndicAccentDb",
    filename="IndicAccentDB.zip",
    repo_type="dataset"
)

print(f"📦 ZIP Downloaded at: {zip_path}")

print("🔓 Extracting ZIP... please wait (2-3 minutes)...")

with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(SAVE_DIR)

print("\n✅ Extraction complete!")
print(f"🎉 Audio files are ready in: {SAVE_DIR}/IndicAccentDb_Audio/")
