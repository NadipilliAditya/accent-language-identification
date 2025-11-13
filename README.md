🎧 Indian Accent / Language Identification App

A Streamlit-based web application that identifies Indian accents/languages from voice input using MFCC features and a machine-learning model.
Users can upload a WAV file or record their own voice, and the system predicts the accent with confidence scores.

⭐ Features
🔊 Voice Input Options

📁 Upload a .wav audio file

🎤 Record voice directly from the browser

🤖 Machine Learning

MFCC feature extraction using Librosa

Classification using a trained Random Forest model

Scaler & Label Encoder included for accurate predictions

🍽️ Bonus Feature:

Accent-based Indian Cuisine Recommendations 😋

🎨 Beautiful UI

Lavender-themed glowing background ✨

Glassmorphism components

Smooth buttons and animated card effects

🏗️ Project Structure
📁 accent-language-identification
 ┣ 📄 app.py
 ┣ 📄 requirements.txt
 ┣ 📄 rf_mfcc_model.joblib
 ┣ 📄 scaler.joblib
 ┣ 📄 label_encoder.joblib
 ┗ 📄 README.md

🚀 How to Run Locally
1️⃣ Create & Activate Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate   # On Windows

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit App
streamlit run app.py

🌐 Deploy on Streamlit Cloud

You can deploy easily by uploading these files:

✔ app.py
✔ requirements.txt
✔ rf_mfcc_model.joblib
✔ scaler.joblib
✔ label_encoder.joblib

Steps:

Go to https://share.streamlit.io/

Click New App

Select your GitHub repo

Choose main branch

Set app.py as the entrypoint

Deploy 🎉

🧠 Model Details

Features: 13 MFCC coefficients

Preprocessing: Mean aggregation

Classifier: Random Forest

Trained on: Indian Accent Dataset

📦 Requirements

All dependencies are listed inside requirements.txt.
Key libraries:

streamlit

numpy

librosa

soundfile

joblib

sounddevice

pandas

👨‍💻 Developer

Aditya Nadipalli


📫 Contact

For improvements, bugs, or contributions, feel free to open issues or pull requests.
