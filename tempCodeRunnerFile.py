import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
from PIL import Image
import os

# App
app = FastAPI()

# Static & Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model
MODEL_PATH = "BrainTumorFixed.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Create upload folder
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------
# IMAGE PREPROCESSING
# ---------------------------

def preprocess_image(image):
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# ---------------------------
# ROUTES
# ---------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    image = Image.open(file_path).convert("RGB")
    image = preprocess_image(image)

    prediction = model.predict(image)[0][0]

    if prediction > 0.5:
        result = "🧠 Tumor Detected"
        confidence = round(prediction * 100, 2)
    else:
        result = "✅ No Tumor Detected"
        confidence = round((1 - prediction) * 100, 2)

    return {
        "result": result,
        "confidence": f"{confidence}%"
    }
