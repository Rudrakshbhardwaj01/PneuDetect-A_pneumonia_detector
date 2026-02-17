import streamlit as st
import numpy as np
import tensorflow as tf
from pathlib import Path
from keras.models import load_model
from keras.applications.densenet import preprocess_input


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pneumonia Detection",
    layout="centered"
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "best_model.keras"
IMG_SIZE = 224


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
@st.cache_resource
def load_trained_model():
    if not MODEL_PATH.exists():
        st.error("Trained model not found. Please train the model first.")
        st.stop()
    return load_model(MODEL_PATH)


def preprocess_image(uploaded_file):
    image = tf.keras.utils.load_img(
        uploaded_file,
        target_size=(IMG_SIZE, IMG_SIZE)
    )
    image = tf.keras.utils.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


def predict(model, image):
    prob = float(model.predict(image)[0][0])
    label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    confidence = prob if label == "PNEUMONIA" else 1 - prob
    return label, confidence


# ---------------------------------------------------------
# App UI
# ---------------------------------------------------------
st.title("🩺 Pneumonia Detection from Chest X-ray")
st.write(
    "Upload a chest X-ray image to get a model prediction. "
    "**This tool is for educational purposes only and is not a medical diagnosis.**"
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded X-ray", use_column_width=True)

    model = load_trained_model()
    image = preprocess_image(uploaded_file)

    with st.spinner("Running inference..."):
        label, confidence = predict(model, image)

    st.subheader("Prediction Result")

    if 0.4 <= confidence <= 0.6:
        st.warning(
            f"⚠️ **Inconclusive result**\n\n"
            f"The model is uncertain.\n\n"
            f"Predicted: **{label}**  \n"
            f"Confidence: **{confidence * 100:.2f}%**"
        )
    else:
        if label == "PNEUMONIA":
            st.error(
                f"🟥 **PNEUMONIA detected**\n\n"
                f"Confidence: **{confidence * 100:.2f}%**"
            )
        else:
            st.success(
                f"🟩 **NORMAL**\n\n"
                f"Confidence: **{confidence * 100:.2f}%**"
            )

    st.caption(
        "⚠️ Disclaimer: This is a machine learning model trained on a limited dataset. "
        "It should NOT be used for clinical diagnosis."
    )
