import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load model (assumes model.h5 is in the same folder)
@st.cache_resource
def load_cnn_model():
    return load_model("model.h5")

model = load_cnn_model()

# Image preprocessing function
def preprocess_image(img):
    img = img.resize((224, 224))  # adjust if different in your training
    img_array = np.array(img)
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit app interface
st.title("Colon Histopathology Cancer Classifier")
st.write("Upload a histopathological image (colon tissue), and we'll predict if it's cancerous.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]

    # Interpret output
    label = "Cancerous" if prediction > 0.5 else "Non-Cancerous"
    confidence = round(prediction * 100, 2) if prediction > 0.5 else round((1 - prediction) * 100, 2)
    st.write(f"### Prediction: {label} ({confidence}% confidence)")
