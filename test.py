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

# Load the saved model
model = load_model('cnn.h5')


# Function to preprocess the image
def preprocess_image(path, target_size):
    print(path)
    img = cv2.imread(path)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0  # normalize pixel values
    return img


# Path to the image you want to predict
image_path = r'./483.jpg'  # replace with the actual path to your image

# Preprocess the image
preprocessed_image = preprocess_image(image_path, (150, 150))

# Reshape the image to match the input shape of the model
input_image = preprocessed_image.reshape(1, 150, 150, 3)

# Make predictions
prediction = model.predict(input_image)

# Convert the prediction to class labels
class_labels = ["Negative", "Positive"]
print(f"prediction value = {prediction[0][0]}")
predicted_class = class_labels[1 if (prediction[0][0] < 0.6) else 0]

# Print the predicted class
print("Predicted class:", predicted_class)