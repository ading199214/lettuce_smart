# app.py
# uvicorn app:app --reload

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse,StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
import numpy as np
import io
from PIL import Image
from openai import OpenAI
import pandas as pd
import json
import requests

app = FastAPI()

# Specify the origins that are allowed to make cross-origin requests
origins = [
    "http://localhost:8080", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows requests from specified origins
    allow_credentials=True,
    allow_methods=["*"],    # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],    # Allows all headers
)
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

@app.get("/insights")
async def insights(request: Request):
    #read history data and manipulate
    df = pd.read_csv('./hyd_data.csv',skiprows=8)
    df_cleaned = df.dropna()
    df_new = df_cleaned.iloc[::200]
    data_json = df_new.to_json(orient='records', lines=True)
    client = OpenAI(api_key="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")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"use below data to summarize the growth process, water, electricity, nutrient solution, etc. How long did it take to complete germination, growth, and maturity. Possible improvements.Based on the above data, given the advertisement, how well the dish absorbs light, moisture, nutrients, and how well the lettuce is. keep it short in 300 words:{data_json}"
            }
        ],
        stream=True,
        max_tokens=300
    )
    def generate():
        for chunk in response:
            # `chunk` will contain the response in small pieces
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    return StreamingResponse(generate(), media_type="text/plain")
        
 
    