# app.py
# uvicorn app:app --reload

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
import numpy as np
import io
from PIL import Image

app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Load the trained model
model = load_model('best_model.keras')

# Define the class labels
class_labels = ['stage1', 'stage2', 'stage3', 'undefined']

# Define the image size
IMAGE_SIZE = (224, 224)

# Define the confidence threshold
CONFIDENCE_THRESHOLD = 0.9  # Adjust this threshold as needed

# Define the main page
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Define the prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        img = img.convert("RGB")
        img = img.resize(IMAGE_SIZE)

        # Preprocess the image
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        probabilities = prediction[0]  # Extract probabilities

        # Get predicted class and confidence
        predicted_class_index = np.argmax(probabilities)
        predicted_prob = probabilities[predicted_class_index]
        confidence = float(predicted_prob)

        predicted_class = class_labels[predicted_class_index]

        # Return the prediction
        return {"predicted_class": predicted_class, "confidence": confidence}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)