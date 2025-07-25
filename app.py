import streamlit as st
import joblib
import urllib.request
import os

@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        url = "https://huggingface.co/JohnAboyeji/colon-cancer-model/resolve/main/coltech_model.pkl"
        urllib.request.urlretrieve(url, "coltech_model.pkl")
    return joblib.load("coltech_model.pkl")

model = load_model()
