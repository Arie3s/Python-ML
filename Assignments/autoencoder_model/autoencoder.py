import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the saved autoencoder model
loaded_model = tf.keras.models.load_model('.')

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input size of the autoencoder
    resized_image = image.resize((width, height))
    # Convert the image to numpy array
    image_array = np.array(resized_image) / 255.0  # Normalize pixel values
    # Expand dimensions to match the shape expected by the model
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Function to decode the image using the autoencoder
def decode_image(image):
    decoded_image = loaded_model.predict(image)
    return decoded_image[0]  # Remove the batch dimension

# Streamlit GUI
st.title('Autoencoder Image Reconstruction')

# Image dropper
st.write("Upload an image:")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Decode button
if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption='Original Image', use_column_width=True)
    
    # Preprocess the uploaded image
    width, height = original_image.size
    processed_image = preprocess_image(original_image)
    
    if st.button('Decode'):
        # Decode the image using the autoencoder
        decoded_image = decode_image(processed_image)
        
        # Convert the decoded image array to PIL image
        decoded_image = Image.fromarray((decoded_image * 255).astype(np.uint8))
        st.image(decoded_image, caption='Decoded Image', use_column_width=True)
