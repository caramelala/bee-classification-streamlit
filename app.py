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
MODEL_PATH = "model_afterAug_FT_logits.h5"
URL = "https://drive.google.com/uc?id=1v7HhpYopAoBMVfnGSC0FIshu4rDX1pku"

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

# PREDIKSI + ENERGY (LOGITS VERSION)
if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # preprocessing
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 1. GET LOGITS
    logits = model.predict(img_array, verbose=0)[0]

    # 2. ENERGY SCORE
    temperature = 1.0
    energy = -temperature * np.log(
        np.sum(np.exp(logits / temperature))
    )

    # 3. SOFTMAX (DISPLAY SAJA)
    probs = tf.nn.softmax(logits).numpy()

    idx = int(np.argmax(probs))
    confidence = float(np.max(probs))

    st.write(f"Confidence: {confidence:.4f}")
    st.write(f"Energy Score: {energy:.4f}")

    # 4. THRESHOLD
    ENERGY_THRESHOLD = -3.1
    CONF_THRESHOLD = 0.85
    
    if confidence < CONF_THRESHOLD or energy > ENERGY_THRESHOLD:
        st.warning("Unknown (Objek di luar lebah)")
    else:
        label = labels[idx]
        st.success(f"Prediksi: {label}")
