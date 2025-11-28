import tensorflow as tf
import os

MODEL_PATH = os.path.join("models", "BrainTumorFixed.h5")

model = tf.keras.models.load_model(MODEL_PATH)

print("✅ Model loaded successfully")
