from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from PIL import Image
import numpy as np

from app.modelloader import model
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


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    image = Image.open(file_path).convert("RGB")
    image = preprocess_image(image)

    prediction = model.predict(image)[0]
    confidence = round(float(max(prediction)) * 100, 2)

    if np.argmax(prediction) == 1:
        result = "🧠 Tumor Detected"
    else:
        result = "✅ No Tumor Detected"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "confidence": f"{confidence}%"
    })
