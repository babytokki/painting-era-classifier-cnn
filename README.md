# Painting Era Classifier 𐔌՞. .՞𐦯
Identifying artistic periods with CNN.

# Overview
This web app uses a TensorFlow Convolutional Neural Network (CNN) deployed in Streamlit to identify whether a painting belongs to Baroque, Medieval, or Renaissance era.
Users can upload a painting image and instantly see:
- Predicted painting era
- Confidence score
- Short description
- Gold-themed UI
- Confidence bar chart (Altair)

# Features
- Custom CNN built & trained from scratch
- Custom styled UI
- Live prediction and confidence chart
- Supports JPG / JPEG / PNG
- Runs locally via Streamlit

# Model
Raw Dataset: Painting Eras Detection Classification Dataset by ArtAncestry
https://share.google/rqdLnLG0PWmTDK0zf

Trained Dataset:
https://drive.google.com/drive/folders/10H5t042JTYi7Mli0XM-xdCBK6hQBXZvv?usp=drive_link

Trained on 3 classes:
- Baroque paintings
- Medieval art
- Renaissance paintings

Preprocessing:
- Resize to 128×128
- Normalization
- CNN architecture (custom)
- Trained in TensorFlow / Keras
- Saved as: painting_era_cnn.h5

# Installation
1️⃣ Clone the repository:
- git clone https://github.com/babytokki/painting-era-classifier.git
- cd painting-era-classifier

2️⃣ Install dependencies:
- pip install -r requirements.txt

▶️ Run the App:
- run main.ipynb
- streamlit run app.py
