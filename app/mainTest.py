import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ✅ Load model
model = load_model('BrainTumorFixed.h5')

# ✅ Image path (change image name if needed)
image_path = r"C:\Users\phoen\OneDrive\Desktop\BrainTumourPredictionAPP\pred\pred0.jpg"
# ✅ Read image
image = cv2.imread(image_path)

if image is None:
    print("❌ Image not found. Check the path.")
    exit()

# ✅ Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ✅ Resize to match training
image = cv2.resize(image, (64, 64))

# ✅ Normalize (same as training)
image = image / 255.0

# ✅ Add batch dimension
image = np.expand_dims(image, axis=0)

print("Input shape:", image.shape)   # Must be (1, 64, 64, 3)

# ✅ Predict
prediction = model.predict(image)
print("Raw prediction:", prediction)

# ✅ Get class index
class_index = np.argmax(prediction)

if class_index == 1:
    print("✅ Tumor Detected")
else:
    print("✅ No Tumor")
