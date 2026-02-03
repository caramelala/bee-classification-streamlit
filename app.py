import streamlit as st

# CSS Watermark
st.markdown("""
<style>
.watermark {
    position: fixed;
    bottom: 10px;
    right: 15px;
    opacity: 0.25;
    font-size: 14px;
    color: gray;
    z-index: 9999;
}
</style>

<div class="watermark">
© Skripsi CNN – VGG16 | Sharla Martiza Yunani
</div>
""", unsafe_allow_html=True)


# CUSTOM CSS / TAMPILAN STREAMLIT
# Mengatur background gradasi kuning-hitam, warna teks, card uploader, dan tombol

import streamlit as st

st.markdown(
    """
    <style>
    /* Background utama */
    .stApp {
        background: linear-gradient(135deg, #FFD700, #000000);
        color: white;
    }

    /* Card / container */
    section[data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 20px;
    }

    /* Judul */
    h1, h2, h3 {
        color: #000000;
        text-align: center;
    }

    /* Teks */
    p, label {
        color: #000000;
    }

    /* Button */
    button {
        background-color: #FFD700 !important;
        color: black !important;
        border-radius: 8px !important;
        font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# DOWNLOAD & LOAD MODEL DARI GOOGLE DRIVE
# Jika file model belum ada, maka otomatis di download kemudian dimuat ke TensorFlow

import gdown
import os

MODEL_PATH = "model_baseline.h5"

URL = "https://drive.google.com/uc?id=1Fon44bP4ey694wiu0wB_u2IhUG_i1nqV"

if not os.path.exists(MODEL_PATH):
    gdown.download(URL, MODEL_PATH, quiet=False)

# STREAMLIT UI & PROSES PREDIKSI GAMBAR

import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
import gdown
from PIL import Image

# DOWNLOAD MODEL DARI DRIVE
MODEL_PATH = "model_baseline.h5"

URL = "https://drive.google.com/uc?id=1Fon44bP4ey694wiu0wB_u2IhUG_i1nqV"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(URL, MODEL_PATH, quiet=False)


# LOAD MODEL
model = tf.keras.models.load_model(MODEL_PATH)

# LOAD CLASS INDICES
with open("class_indices.json") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

# STREAMLIT 
st.title("🐝 Bee Classification App")

st.markdown(
    """
    <p style='text-align:center; font-size:16px; color:black;'>
        <b>Sharla Martiza Yunani</b><br>
        Universitas Lampung
    </p>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload gambar lebah",
    type=["jpg", "jpeg", "png"]
)

# PREPROCESS & PREDICT
if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)

    idx = np.argmax(pred)

    st.success(f"Prediksi: **{labels[idx]}**")
