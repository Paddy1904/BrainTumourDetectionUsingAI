from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from PIL import Image
import numpy as np

from app.modelloader import get_model
from app.utils import preprocess_image

app = FastAPI()

# Static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get('/health')
def health():
    return {"status": "ok"}

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    image = Image.open(file_path).convert("RGB")
    image = preprocess_image(image)

    # Prediction handling robust to binary or multiclass outputs
    try:
        model = get_model()
        prediction_raw = model.predict(image)
    except Exception as e:
        # Log error and return a friendly error page
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": f"Model prediction error: {str(e)}",
            "confidence": "0%"
        })
    prediction = np.asarray(prediction_raw).squeeze()

    if prediction.ndim == 0:
        # Single probability output (binary)
        prob = float(prediction)
        label = 1 if prob > 0.5 else 0
        confidence = round(prob * 100, 2) if label == 1 else round((1 - prob) * 100, 2)
    else:
        # Multiclass
        label = int(np.argmax(prediction))
        confidence = round(float(np.max(prediction)) * 100, 2)
    result = "🧠 Tumor Detected" if label == 1 else "✅ No Tumor Detected"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "confidence": f"{confidence}%"
    })
