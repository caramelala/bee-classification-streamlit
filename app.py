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
MODEL_PATH = "model_afterAug_FT.h5"
URL = "https://drive.google.com/uc?id=1I7H0W-BNJEhlnUsdjyltoummMShWvfug"
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(URL, MODEL_PATH, quiet=False)
# LOAD MODEL
model = tf.keras.models.load_model(MODEL_PATH)
# LOAD CLASS INDICES
with open("class_indices.json") as f:
    class_indices = json.load(f)
labels = {v: k for k, v in class_indices.items()}

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
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

# PREPROCESS & PREDICT (ENERGY-BASED OOD)
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Softmax prediction
    probs = model.predict(img_array, verbose=0)[0]

    # Logits prediction
    logits = logits_model.predict(img_array, verbose=0)[0]

    # ENERGY SCORE (log-sum-exp)
    energy = -np.log(np.sum(np.exp(logits)))

    # THRESHOLD ENERGY
    ENERGY_THRESHOLD = 4.5  

    if energy > ENERGY_THRESHOLD:
        st.warning("Prediksi: **Unknown (Objek di luar kelas lebah)**")
    else:
        idx = np.argmax(probs)
        st.success(f"Prediksi: **{labels[idx]}**")
