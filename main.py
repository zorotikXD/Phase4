import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os

# Load Model & Scaler
MODEL_PATH = os.path.abspath("./autoencoder_model.h5")
SCALER_PATH = os.path.abspath("./scaler.pkl")

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict(input_data):
    feature_names = ["customer_id", "products_purchased", "complains", "money_spent"]
    input_df = pd.DataFrame([input_data], columns=feature_names)
    scaled_input = scaler.transform(input_df)
    encoded_data = model.predict(scaled_input)
    return np.array(encoded_data)



# Streamlit UI
st.title("Customer Market Segmentation Prediction")
customer_id = st.number_input("Customer ID", min_value=1000000, max_value=9999999, value=1000661)
products_purchased = st.number_input("Products Purchased", min_value=0, max_value=100, value=1)
complains = st.number_input("Complains", min_value=0, max_value=10, value=0)
money_spent = st.number_input("Money Spent", min_value=0.0, value=260.0)

if st.button("Predict"):
    input_data = [customer_id, products_purchased, complains, money_spent]
    prediction = predict(input_data)
    st.write("Encoded Output:", prediction)
