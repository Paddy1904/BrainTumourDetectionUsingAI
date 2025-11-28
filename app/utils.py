import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image):
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image
