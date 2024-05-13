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
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500"],  # Allow requests from this origin
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
# Load your pre-trained model
try:
    model = load_model('cnn.h5')
except Exception as e:
   print("----------------------------------------------------")
   print(e)
   print("----------------------------------------------------")

def preprocess_image(image_bytes, target_size):
    imm = Image.open(BytesIO(image_bytes))
    img = cv2.imread(imm)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0 # normalize pixel values
    return img

@app.post("/predict/")
async def predict(image_file: UploadFile = File(...)):
    # Read the image file as bytes
    print(f"PRINTING THE IMAGE FILE {image_file}")
    image_bytes = await image_file.read()
    preprocessed_image = preprocess_image(image_bytes, (150, 150))
    input_image = preprocessed_image.reshape(1, 150, 150, 3)
    prediction = model.predict(input_image)
    class_labels = ["Negative", "Positive"]
    predicted_class = class_labels[1 if (prediction[0][0] < 0.6) else 0]
    
    # Perform object detection with YOLOv8
    print(predicted_class)
    # Optionally, you can return the prediction result
    return {"prediction": predicted_class}