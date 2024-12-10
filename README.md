# P1-Image-Classification-Using-ML
1. Setup

Google Colab: Start a new notebook in Google Colab.
Libraries: Import the necessary libraries.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import io
2. Pre-trained Model

We'll use MobileNetV2 for this example.

model = keras.applications.MobileNetV2(weights='imagenet')
3. Create Upload Function

def upload_image():
  uploaded = files.upload()
  for fn in uploaded.keys():
    img_path = io.BytesIO(uploaded[fn])
    return img_path
4. Preprocess Image

def preprocess_image(img_path):
  img = image.load_img(img_path, target_size=(224, 224))
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array /= 255.0
  return img_array
5. Make Predictions

def make_prediction(img_array):
  predictions = model.predict(img_array)
  decoded_predictions = decode_predictions(predictions, top=3)
  return decoded_predictions
6. Display Results

def display_results(decoded_predictions):
  for pred in decoded_predictions[0]:
    print(f"{pred[1]}: {pred[2]*100:.2f}%")
7. Combine into a Single Function

def classify_image():
  img_path = upload_image()
  if img_path:
    img_array = preprocess_image(img_path)
    predictions = make_prediction(img_array)
    display_results(predictions)
  else:
    print("No image uploaded.")
Implementation Instructions

Copy and paste the code into your Google Colab notebook.
Run the code cells.
Call the classify_image() function.
When prompted, upload an image from your local machine.
The code will preprocess the image, make predictions using the pre-trained model, and display the top predicted classes with probabilities.
