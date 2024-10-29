from flask import Flask, request, render_template
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load the pre-trained model
model = load_model('signature_model.h5')  # Ensure you save your trained model as 'signature_model.h5'

# Constants
IMG_SIZE = (128, 128)

# Function to predict signature authenticity
def predict_signature(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE) / 255.0  # Resize and normalize
    img = img.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)  # Reshape for the model
    prediction = model.predict(img)
    return "genuine" if prediction[0] > 0.5 else "forged"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    # Save the uploaded file to a temporary location
    temp_path = os.path.join('static', 'uploads', file.filename)
    file.save(temp_path)

    # Predict signature authenticity
    result = predict_signature(temp_path)
    
    # Clean up the saved file (optional)
    os.remove(temp_path)

    return result

if __name__ == '__main__':
    app.run(debug=True)
