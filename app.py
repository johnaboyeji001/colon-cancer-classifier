import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model once
model = tf.keras.models.load_model("colon_cancer_cnn_model.h5")

# Set title
st.title("Colon Cancer Slide Classifier")
st.write("Upload a histopathology image to predict if it's cancerous.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_height, img_width = 224, 224  # Replace with the dimensions used in training
    img = image.resize((img_width, img_height))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "Cancerous" if prediction > 0.5 else "Non-Cancerous"
    confidence = round(prediction if prediction > 0.5 else 1 - prediction, 2)

    st.write(f"### Prediction: {label}")
    st.write(f"Confidence: {confidence}")
