import io
import numpy as np
from PIL import Image
from tensorflow.keras.applications import mobilenet_v2, efficientnet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras import Sequential

MODEL_CACHE = {}

DISEASE_CLASSES = [
    "Early Blight",
    "Powdery Mildew",
    "Leaf Blast",
    "Healthy",
]

def load_tf_model(model_type="efficientnet"):
    key = model_type.lower()
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    if key == "mobilenet":
        base = mobilenet_v2.MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling="avg")
        preprocess = mobilenet_v2.preprocess_input
    else:
        base = efficientnet.EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling="avg")
        preprocess = efficientnet.preprocess_input

    model = Sequential([base])
    MODEL_CACHE[key] = (model, preprocess)
    return MODEL_CACHE[key]


def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_disease(file_bytes, model, preprocess_input):
    arr = preprocess_image(file_bytes)
    arr = preprocess_input(arr)
    features = model.predict(arr, verbose=0)
    score = float(np.mean(features))

    # deterministic mapping for mock disease classes
    class_idx = int((score * len(DISEASE_CLASSES)) % len(DISEASE_CLASSES))
    confidence = min(99, max(70, int((score % 1) * 100)))

    return {
        "disease": DISEASE_CLASSES[class_idx],
        "confidence": confidence,
    }
