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

# DOWNLOAD & LOAD MODEL
MODEL_PATH = "model_with_unknown.h5"
URL = "https://drive.google.com/uc?id=1DnRyCyEQgPQBEcNx28QkwNcIEhw7eQrS"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(URL, MODEL_PATH, quiet=False, fuzzy=True)

model = tf.keras.models.load_model(MODEL_PATH)

# LOAD CLASS INDICES
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# balik mapping: index → label
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
    "Upload gambar",
    type=["jpg", "jpeg", "png"]
)

# PREPROCESS & PREDICT
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array, verbose=0)
    idx = int(np.argmax(pred))

    label = labels.get(idx, "Unknown")

    # Kalau kamu pakai kelas 'objek' sebagai unknown
    if label.lower() in ["objek", "unknown", "non_lebah"]:
        st.warning("Prediksi: **Unknown (Objek di luar lebah)**")
    else:
        st.success(f"Prediksi: **{label}**")
