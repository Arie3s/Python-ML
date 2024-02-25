import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the denoising autoencoder model
model = tf.keras.models.load_model('.')

# Function to preprocess the image
def preprocess_image(image):
    resized_image = image.resize((28, 28))  # Resize the image
    grayscale_image = resized_image.convert('L')  # Convert to grayscale
    image_array = np.array(grayscale_image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    return image_array

# Function to denoise the image using the autoencoder
def denoise_image(image):
    denoised_image = model.predict(image)
    return denoised_image

# Streamlit GUI
st.title('Denoising Autoencoder')

# Image dropper
st.write("Upload an image:")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Decode button
if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption='Original Image', use_column_width=True)
    
    # Preprocess the uploaded image
    processed_image = preprocess_image(original_image)
    
    if st.button('Denoise'):
        # Denoise the image using the autoencoder
        denoised_image = denoise_image(processed_image)
        
        # Convert the denoised image array to PIL image
        denoised_image = denoised_image.squeeze() * 255
        denoised_image = Image.fromarray(denoised_image.astype(np.uint8))
        st.image(denoised_image, caption='Denoised Image', use_column_width=True)
