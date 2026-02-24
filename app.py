import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
import gdown
from PIL import Image

# CSS WATERMARK
st.markdown("""
<style>
.watermark {
    position: fixed;
    bottom: 10px;
    left: 15px;
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

# DOWNLOAD MODEL DARI DRIVE
MODEL_PATH = "model_with_unknown.h5"
URL = "https://drive.google.com/uc?id=1DnRyCyEQgPQBEcNx28QkwNcIEhw7eQrS"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(URL, MODEL_PATH, quiet=False)
# LOAD MODEL
model = tf.keras.models.load_model(MODEL_PATH)

# MODEL TANPA SOFTMAX (LOGITS)
logits_model = tf.keras.Model(
    inputs=model.input,
    outputs=model.layers[-1].input
)

# LOAD CLASS INDICES
with open("class_indices.json") as f:
    class_indices = json.load(f)

labels = {int(v): k for k, v in class_indices.items()}

# STREAMLIT UI
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
idx = int(np.argmax(pred))

if idx not in labels:
    st.warning("Prediksi: **Unknown (label tidak dikenali model)**")
else:
    st.success(f"Prediksi: **{labels[idx]}**")
