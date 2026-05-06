import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
import gdown
from PIL import Image

# =========================
# CSS WATERMARK
# =========================
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
©️ Skripsi CNN – VGG16 | Sharla Martiza Yunani
</div>
""", unsafe_allow_html=True)

# =========================
# DOWNLOAD MODEL
# =========================
MODEL_PATH = "model_afterAug_FT_logits.h5"
URL = "https://drive.google.com/uc?id=1T7fyazI0JiyJM9yBcdZ2P3dsV2hVB3Re"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(URL, MODEL_PATH, quiet=False)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# LOAD LABEL
# =========================
with open("class_indices.json") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

# =========================
# UI
# =========================
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

# =========================
# PREDIKSI
# =========================
if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict logits
    logits = model.predict(img_array, verbose=0)[0]

    # Temperature
    T = 1.0

    # Energy Score
    energy = -T * np.log(
        np.sum(np.exp(logits / T))
    )

    # Softmax
    probs = tf.nn.softmax(logits).numpy()
    idx = int(np.argmax(probs))
    confidence = float(np.max(probs))

    st.write(f"Confidence: {confidence:.4f}")
    st.write(f"Energy Score: {energy:.4f}")

    # Threshold
    CONF_THRESHOLD = 0.85
    ENERGY_THRESHOLD = -17

    # Decision
    if confidence < CONF_THRESHOLD or energy > ENERGY_THRESHOLD:
        st.warning("Prediksi: **Unknown (Objek di luar lebah)**")
    else:
        st.success(f"Prediksi: **{labels[idx]}**")

    # Debug info
    st.write("CONF_THRESHOLD:", CONF_THRESHOLD)
    st.write("ENERGY_THRESHOLD:", ENERGY_THRESHOLD)
    st.write("Energy > Threshold ?", energy > ENERGY_THRESHOLD)
