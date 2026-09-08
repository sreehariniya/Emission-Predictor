import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import PolynomialFeatures

# Page configuration
st.set_page_config(page_title="Sustainability & CO2 Emissions Predictor", page_icon="🌱", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("polynomialRegModel.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("sustainability_data.csv")

try:
    model = load_model()
    df = load_data()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model or dataset: {e}")

st.title("🌱 CO₂ Emissions Predictor")
st.write("Estimate total **CO₂ Emissions** based on key energy and economic indicators.")

if model_loaded:
    st.subheader("Input Indicators")
    
    # Input sliders/fields with default values based on dataset averages
    energy = st.number_input(
        "Energy Consumption", 
        min_value=0.0, 
        max_value=2000.0, 
        value=float(df["Energy_Consumption"].mean()), 
        step=5.0
    )
    
    renewable = st.number_input(
        "Renewable Energy Percentage (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=float(df["Renewable_Percentage"].mean()), 
        step=0.5
    )
    
    gdp = st.number_input(
        "GDP", 
        min_value=0.0, 
        max_value=10000.0, 
        value=float(df["GDP"].mean()), 
        step=50.0
    )

    if st.button("Predict CO₂ Emissions"):
        # Apply Polynomial Transformation (Degree 2, include_bias=True)
        raw_input = np.array([[energy, renewable, gdp]])
        poly = PolynomialFeatures(degree=2, include_bias=True)
        transformed_input = poly.fit_transform(raw_input)
        
        # Predict target
        prediction = model.predict(transformed_input)[0]
        
        st.markdown("---")
        st.metric(label="Predicted CO₂ Emissions", value=f"{prediction:.2f}")

    # Dataset preview section
    with st.expander("View Historical Sustainability Data"):
        st.dataframe(df)
