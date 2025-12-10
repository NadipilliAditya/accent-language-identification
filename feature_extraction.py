import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, HubertModel

# Load HuBERT model only once
extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

def extract_hubert_embeddings(audio):
    inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = hubert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
