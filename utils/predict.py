from pathlib import Path
import numpy as np
import tensorflow as tf
from keras.applications.densenet import preprocess_input
from keras.models import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "best_model.keras"


def load_image(image_path: str, img_size: int = 224) -> np.ndarray:
    """
    Load and preprocess a single image for prediction.
    """
    image = tf.keras.utils.load_img(
        image_path,
        target_size=(img_size, img_size)
    )
    image = tf.keras.utils.img_to_array(image)
    image = np.expand_dims(image, axis=0)  # (1, H, W, C)
    image = preprocess_input(image)
    return image


def predict_image(image_path: str) -> dict:
    """
    Predict pneumonia from a single chest X-ray image.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Train the model first.")

    model = load_model(MODEL_PATH)

    image = load_image(image_path)
    prob = float(model.predict(image)[0][0])

    label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    confidence = prob if label == "PNEUMONIA" else 1 - prob

    return {
        "label": label,
        "confidence": round(confidence * 100, 2)
    }


if __name__ == "__main__":
    # Simple CLI test
    test_image = input("Enter path to chest X-ray image: ").strip()
    result = predict_image(test_image)
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']}%")
