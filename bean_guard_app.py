# Import necessary libraries
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the pre-trained model
@st.cache_resource
def load_trained_model():
    model_path = r"E:\Horizon\4th Year\1st Semester\Final Project\Colab\plant_disease_model.h5"  
    return load_model(model_path)

model = load_trained_model()

# Class labels (update these with your actual class names)
class_labels = ["ALS", "Blight", "Fresh_Leaf", "Mosaic_Virus", "Rust"] 

# Preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((128, 128))  # Resize to match model input size
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit app title and description
st.title("Plant Disease Classification App 🌱")
st.write("Upload an image of a plant leaf to detect its disease.")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Process and predict
        img = load_img(uploaded_file)  # Load the image
        img_array = preprocess_image(img)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        predicted_label = class_labels[predicted_class]

        # Display prediction
        st.write(f"Predicted Class: **{predicted_label}**")

    except Exception as e:
        st.error(f"Error processing the image: {e}")
