import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image

st.set_page_config(page_title="Painting Era Classifier", page_icon="🎨", layout="centered")

st.markdown("""
<style>

/* Page background */
body {
    background-color: #faf7f2;
}
[data-testid="stAppViewContainer"] {
    background-color: #faf7f2;
}
[data-testid="stAppViewContainer"]::before {
    background-color: #faf7f2;
}

/* ----------- Global Card Theme ----------- */
.result-card, .upload-card {
    padding: 25px;
    border-radius: 18px;
    background-color: #fffdf9;
    border: 1px solid #e8e3dc;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}

/* ----------- Title ----------- */
h1 {
    font-weight: 800 !important;
    color: #D4AF37 !important;
}

/* ----------- Era Badge ----------- */
.era-badge {
    display: inline-block;
    background: linear-gradient(135deg, #e4c26d, #cba74d);
    color: white;
    padding: 8px 18px;
    border-radius: 999px;
    font-weight: bold;
    font-size: 16px;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
}

/* ----------- Section Titles ----------- */
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 30px;
}

/* ----------- Uploader Styling ----------- */
.stFileUploader {
    padding: 22px !important;
    background-color: #fffdf9 !important;
    border-radius: 18px !important;
    border: 1px solid #e8e3dc !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}

.stFileUploader label {
    font-weight: 600 !important;
    color: #4a4a4a !important;
}

/* ----------- Uploaded Image Container ----------- */
.uploaded-image-card {
    border-radius: 22px;
    padding: 12px;
    background: #ffffff;
    border: 1px solid #e8e3c6;
    box-shadow: 0 4px 14px rgba(0,0,0,0.07);
    margin-top: 15px;
    width: fit-content;
    margin-left: auto;
    margin-right: auto;
}

.uploaded-caption {
    text-align: center;
    font-size: 0.9rem;
    padding-top: 0.4rem;
    color: #6a6a6a;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='
    text-align: center;
    margin-top: 20px;
    color: #D4AF37 !important;
'>
    Painting Era Classifier
</h1>
""", unsafe_allow_html=True)


# ---------------------------- Load Model (.h5) ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("painting_era_cnn.h5")

model = load_model()


# ---------------------------- Preprocessing ----------------------------
IMG_SIZE = (128, 128)

def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# ---------------------------- Class Info ----------------------------
class_names = {
    "Baroque paintings": {
        "range": "1600–1750 CE",
        "description": "Known for dramatic lighting (chiaroscuro), rich deep colors, high contrast, and emotional intensity."
    },
    "Medieval art": {
        "range": "500–1400 CE",
        "description": "Characterized by religious themes, flat perspective, symbolic imagery, and strong outlines with gold accents."
    },
    "Renaissance paintings": {
        "range": "1400–1600 CE",
        "description": "Focused on realism, perspective, anatomical accuracy, and classical harmony."
    }
}

class_list = list(class_names.keys())


# ---------------------------- File Upload ----------------------------
uploaded_file = st.file_uploader("Upload a painting", type=["jpg", "jpeg", "png"])

if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")

    # Image display
    st.markdown('<div class="uploaded-image-card">', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="uploaded-caption">Uploaded Image</div>', unsafe_allow_html=True)

    # Prediction
    img_array = preprocess_image(img)
    preds = model.predict(img_array)[0]

    predicted_idx = int(np.argmax(preds))
    predicted_key = class_list[predicted_idx]
    confidence = float(np.max(preds))

    info = class_names[predicted_key]

    # ---------------------------- Result Card ----------------------------
    st.markdown('<div class="section-title">Classification Results</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card">
        <div class="era-badge">{predicted_key}</div>
        <h4 style="margin-top: 5px;">{info['range']}</h4>
        <p style="font-size: 17px; line-height: 1.5; margin-top: 10px;">
            {info['description']}
        </p>
        <p style="font-size: 18px; margin-top: 15px;">
            <b>Confidence:</b> {confidence:.2f}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------- Confidence Chart ----------------------------
    st.markdown('<div class="section-title">Confidence Analysis</div>', unsafe_allow_html=True)

    df = pd.DataFrame({
        "Class": class_list,
        "Confidence": preds
    })

    chart = (
        alt.Chart(df)
        .mark_bar(color="#D4AF37")
        .encode(
            x=alt.X("Class", sort=None),
            y="Confidence"
        )
        .properties(height=300)
    )

    st.altair_chart(chart, use_container_width=True)
