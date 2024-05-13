import uvicorn
from fastapi import FastAPI, File, UploadFile
import joblib
from PIL import Image
import numpy as np
from io import BytesIO
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from ultralytics import YOLO
import cv2
from fastapi.responses import JSONResponse
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500"],  # Allow requests from this origin
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
# Load your pre-trained model
import os

UPLOAD_DIR = "uploads"  # Define the directory to save uploaded files

# Create the uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

try:
    model = load_model('cnn.h5')
except Exception as e:
   print("----------------------------------------------------")
   print(e)
   print("----------------------------------------------------")

def preprocess_image(path, target_size):
    print(path)
    img = cv2.imread(path)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0  # normalize pixel values
    return img

@app.post("/predict/")
async def predict(image_file: UploadFile = File(...)):
    # Read the image file as bytes
    file_path = os.path.join(UPLOAD_DIR, image_file.filename)
    with open(file_path, "wb") as f:
        f.write(await image_file.read())
    preprocessed_image = preprocess_image(file_path, (150, 150))
    input_image = preprocessed_image.reshape(1, 150, 150, 3)
    prediction = model.predict(input_image)
    class_labels = ["Negative", "Positive"]
    # print(f"prediction value = {prediction[0][0]}")
    predicted_class = class_labels[1 if (prediction[0][0] < 0.6) else 0]
    print("Predicted class:", predicted_class)
    response_data = {"predicted_class": predicted_class}
    print(response_data)
    return JSONResponse(content=response_data)