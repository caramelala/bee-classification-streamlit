import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown
from PIL import Image

# WATERMARK
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

# DOWNLOAD MODEL
MODEL_PATH = "model_afterAug_FT.h5"
URL = "https://drive.google.com/uc?id=1I7H0W-BNJEhlnUsdjyltoummMShWvfug"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(URL, MODEL_PATH, quiet=False)

# LOAD MODEL
model = tf.keras.models.load_model(MODEL_PATH)

labels = {
    0: "Geniotrigona thoracica",
    1: "Tetragonula laeviceps",
    2: "Tetragonula testaceitarsis",
    3: "Tetrigona binghami",
}

# UI
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

# PREDIKSI + ENERGY
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array, verbose=0)

    # ENERGY CALCULATION
    CONF_THRESHOLD = 0.75

    # Threshold (HARUS DI-TUNING)
    ENERGY_THRESHOLD = 1.5

    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    st.write(f"Confidence: {confidence:.4f}")
    st.write(f"Energy Score: {energy:.4f}")

    # DECISION
    CONF_THRESHOLD = 0.90
    
    idx = int(np.argmax(probs))
    confidence = float(np.max(probs))
    
    if confidence < CONF_THRESHOLD:
        st.warning("Prediksi: **Unknown (Objek di luar lebah)**")
    else:
        label = labels.get(idx, "Unknown")
        st.success(f"Prediksi: **{label}**")
