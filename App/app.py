# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Load model, scaler, and encoder ---
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "rf_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
encoder_path = os.path.join(BASE_DIR, "encoder.pkl")
training_columns_path = os.path.join(BASE_DIR, "training_columns.pkl")  # saved df.columns

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)
training_columns = joblib.load(training_columns_path)  # df.columns from training

# --- Streamlit UI ---
st.title("Car Price Prediction 🚗💰")

# Sidebar inputs
brand_name = st.sidebar.text_input("Car Name (e.g., Audi A4)")
km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, step=1000)
engine = st.sidebar.number_input("Engine (CC)", min_value=500, step=50)
max_power = st.sidebar.number_input("Max Power (bhp)", min_value=30, step=5)
torque = st.sidebar.number_input("Torque (Nm)", min_value=50, step=10)
mileage = st.sidebar.number_input("Mileage (kmpl)", min_value=5, step=1)

# --- Prepare input DataFrame ---
input_df = pd.DataFrame([[brand_name, km_driven, engine, max_power, torque, mileage]],
                        columns=['name', 'km_driven', 'engine', 'max_power', 'torque', 'mileage'])

# Extract brand and drop 'name'
input_df['brand'] = input_df['name'].str.split().str[0]
input_df.drop('name', axis=1, inplace=True)

# One-hot encode brand using saved encoder
encoded_input = encoder.transform(input_df[['brand']])
encoded_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(['brand']), index=input_df.index)

# Combine with original input and drop 'brand' column
input_df = pd.concat([input_df.drop('brand', axis=1), encoded_df], axis=1)

# Ensure all training columns are present
missing_cols = set(training_columns) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0
# Reorder columns to match training
input_df = input_df[training_columns]

# Scale features
input_scaled = scaler.transform(input_df)

# --- Predict button ---
if st.button("Predict Price"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"Estimated Selling Price: ₹ {prediction:,.2f}")
