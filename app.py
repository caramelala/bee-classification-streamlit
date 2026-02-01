import gdown
import os

MODEL_PATH = "model_baseline.h5"

URL = "https://drive.google.com/uc?id=1Fon44bP4ey694wiu0wB_u2IhUG_i1nqV"

if not os.path.exists(MODEL_PATH):
    gdown.download(URL, MODEL_PATH, quiet=False)



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
