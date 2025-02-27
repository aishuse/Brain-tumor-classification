import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("cnn_model.h5")

# Define categories
Categories = ['glioma', 'pituitary', 'notumor', 'meningioma']

def preprocess_image(image):
    """Preprocess the uploaded image for prediction."""
    image = np.array(image.convert("L"))  # Convert to grayscale
    image = cv2.resize(image, (150, 150))  # Resize to match model input
    image = image.reshape(1, 150, 150, 1)  # Reshape for model
    image = image / 255.0  # Normalize pixel values
    return image

# Streamlit UI
st.title("Brain Tumor Classification")
st.write("Upload an MRI image to classify the type of brain tumor.")

uploaded_file = st.file_uploader("Choose an MRI Image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    pred = model.predict(processed_image)
    ind = pred.argmax(axis=1)
    result = Categories[ind.item()]
    
    # Display result
    st.write("### Prediction: ", result)
