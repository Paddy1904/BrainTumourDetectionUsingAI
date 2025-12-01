import tensorflow as tf
from pathlib import Path

# Find the model file in a few common locations (root 'models/', 'templates/models/')
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_CANDIDATES = [
    BASE_DIR / "models" / "BrainTumorFixed.h5",
    BASE_DIR / "templates" / "models" / "BrainTumorFixed.h5",
    BASE_DIR / "BrainTumourDetectionUsingAI" / "templates" / "models" / "BrainTumorFixed.h5",
]

MODEL_PATH = None
for p in MODEL_CANDIDATES:
    if p.exists():
        MODEL_PATH = str(p)
        break

if MODEL_PATH is None:
    raise FileNotFoundError(
        "Model file 'BrainTumorFixed.h5' not found. Looked in: " + ", ".join(str(x) for x in MODEL_CANDIDATES)
    )

model = None

def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    return model
