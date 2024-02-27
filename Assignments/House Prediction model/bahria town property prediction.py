# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 00:34:56 2024

@author: walit
"""

import streamlit as st
import joblib

# Load the trained model
model = joblib.load('house_pred.pkl')

# Define a function to predict house price based on area
def predict_price(area):
    # Assume other features are fixed (you may need to modify this based on your model)
    # Convert area to float
    area = float(area)
    # Predict the price
    price = model.predict([[area]])
    return price[0]

# Streamlit app
def main():
    # Set app title
    st.title('Bahria Town Karachi Property Price Predictor')

    # Add a text input for the area
    area = st.text_input('Enter the area (in square feet):')

    # Add a button to make predictions
    if st.button('Predict Price'):
        # Validate input
        if area:
            # Predict the price
            price = predict_price(area)
            # Display the predicted price
            st.success(f'The estimated price of the property is PKR {price:,.2f}')
        else:
            st.warning('Please enter the area.')

# Run the app
if __name__ == '__main__':
    main()
