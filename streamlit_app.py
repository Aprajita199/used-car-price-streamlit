import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('car_price_model.h5')

# Load preprocessor
with open('scaler.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# UI
st.title('Used Car Price Predictor')

age = st.number_input('Age of the car (years)', min_value=0, max_value=30, step=1)
mileage = st.number_input('Mileage (in miles)', min_value=0.0, step=100.0)
fuel_type = st.selectbox('Fuel Type', ['Gasoline', 'Hybrid', 'E85 Flex Fuel'])

if st.button('Predict Price'):
    input_df = pd.DataFrame([[age, mileage, fuel_type]], columns=['age', 'milage', 'fuel_type'])
    processed = preprocessor.transform(input_df)
    prediction = model.predict(processed)
    st.success(f"Estimated Price: ₹{round(prediction[0][0], 2):,}")
