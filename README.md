# PneuDetect – Chest X-ray Pneumonia Classifier

PneuDetect is a machine learning web application that classifies chest X-ray images as **PNEUMONIA**, **NORMAL**, or **UNCERTAIN** using a convolutional neural network.

This project demonstrates an end-to-end ML workflow including data preprocessing, transfer learning, model training, and deployment with Streamlit.

---

## Features

- DenseNet-based transfer learning model
- Binary classification with calibrated uncertainty handling
- Simple web interface built using Streamlit
- Input validation to reduce misleading predictions
- Designed for fast deployment on Streamlit Cloud

---

## Tech Stack

- Python
- TensorFlow (CPU)
- Keras
- Streamlit
- NumPy
- Pillow

---

## Usage

1. Upload a chest X-ray image (JPEG or PNG)
2. The model analyzes the image
3. The app displays:
   - Predicted class (PNEUMONIA / NORMAL / UNCERTAIN)
   - Confidence score

---

## Deployment

This application is deployed using **Streamlit Community Cloud** directly from this GitHub repository.

---

## Limitations

- Trained on a limited chest X-ray dataset
- Not robust to non–X-ray or out-of-distribution images
- Model predictions may be uncertain for ambiguous cases
- Not clinically validated

---

## Disclaimer

This project is provided **strictly for educational and demonstration purposes**.

It is **NOT** a medical device and **MUST NOT** be used for diagnosis, treatment, or clinical decision-making.  
All outputs represent predictions from a machine learning model, not medical conclusions.

---

## License

For educational and non-commercial use only.
