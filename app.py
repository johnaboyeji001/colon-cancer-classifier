import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import requests
import os

# URL of the model on Hugging Face
MODEL_URL = "https://huggingface.co/JohnAboyeji/colon-cancer-model/resolve/main/colon_cancer_cnn_model.h5"

MODEL_PATH = "model.h5"

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
    return load_model(MODEL_PATH)

model = download_model()

def preprocess_image(img):
    img = img.resize((224, 224))  # Change based on your training
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title("Colon Cancer Histopathology Classifier")
st.write("Upload an image to predict whether it's cancerous or not.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]
    label = "Cancerous" if prediction > 0.5 else "Non-Cancerous"
    confidence = round(prediction * 100, 2) if prediction > 0.5 else round((1 - prediction) * 100, 2)
    st.write(f"### Prediction: {label} ({confidence}% confidence)")
