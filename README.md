# 🎙️ Native Language & Accent Identification App
### Powered by **MFCC + HuBERT Embeddings** | Built with **Streamlit**

This project is a **Machine Learning & Speech Processing Web Application** that identifies the **native language (L1)** of Indian English speakers using acoustic and self-supervised speech features.  
It also provides a **personalized cuisine recommendation** based on the detected accent.

---

## 🌟 Project Objective
The system aims to automatically classify the **native language** of a speaker who is speaking English, purely from their accent patterns.

It uses:
- **MFCCs** (traditional features)
- **HuBERT embeddings**
- **Machine Learning models** (Random Forest)

---

## 🧠 Features
- Upload `.wav` files  
- Record audio using the microphone  
- Extract MFCC features  
- Predict native language or accent  
- Show prediction confidence  
- Recommend cuisine based on accent  
- Modern Streamlit UI  

---

## 🗂️ Dataset
**IndicAccentDB – Indian English Accent Dataset**  
Contains speech samples for native languages such as Hindi, Tamil, Telugu, Malayalam, Kannada, Bengali, Gujarati, Marathi, and Punjabi.

Dataset Link: https://huggingface.co/datasets/DarshanaS/IndicAccentDb

---

## 🧩 Technology Stack
| Component | Usage |
|----------|--------|
| Python | Core programming |
| Streamlit | Web UI |
| Librosa | Audio processing |
| Scikit-learn | ML Model |
| HuBERT | Embeddings |
| Streamlit Cloud | Deployment |

---

## 🔍 Key Research Elements
### 1️⃣ Native Language Identification
- MFCC vs HuBERT comparison  
- Layer-wise analysis  
- Experiments with CNN, BiLSTM, and Transformer models  
- Hyperparameter tuning  

### 2️⃣ Age Generalization
Train on adults → Test on children.

### 3️⃣ Word vs Sentence Analysis
Compare how different speech units preserve accent cues.

### 4️⃣ Accent-Aware Cuisine Recommender
Example:
- Malayalam accent → Appam, Puttu  
- Punjabi accent → Butter Chicken  

---

## 📁 Repository Contents
- `app.py`  
- `model.pkl`  
- `feature_extraction.py`  
- `utils.py`  
- `requirements.txt`  
- `README.md`  

---

## 🚀 Run Locally
```
pip install -r requirements.txt
streamlit run app.py
```

---

## 🎯 Expected Outcomes
- Functional NLI model  
- MFCC & HuBERT performance comparison  
- Layer-wise HuBERT insights  
- Age generalization analysis  
- Sentence vs word evaluation  
- Web app with cuisine recommendation  

---

## 📚 Conceptual Background
Accents arise due to:
- Native language phonetic influence  
- Vowel and consonant articulation differences  
- Prosodic rhythm  

These can be captured using MFCCs and HuBERT embeddings.
