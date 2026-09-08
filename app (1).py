
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Sustainability Prediction",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Sustainability Prediction App")
st.write("Predict CO₂ emissions using sustainability data.")

# Load dataset
data = pd.read_csv("sustainability_data.csv")

# Display dataset
st.subheader("📊 Sustainability Data")
st.dataframe(data)

# Load model
try:
    with open("polynomialRegModel.pkl", "rb") as file:
        model = pickle.load(file)

    st.success("Polynomial Regression Model Loaded Successfully!")

    st.subheader("🔮 Make a Prediction")

    energy = st.number_input(
        "Energy Consumption",
        min_value=0.0,
        value=500.0
    )

    renewable = st.number_input(
        "Renewable Percentage",
        min_value=0.0,
        max_value=100.0,
        value=25.0
    )

    gdp = st.number_input(
        "GDP",
        min_value=0.0,
        value=1500.0
    )

    if st.button("Predict CO₂ Emissions"):

        input_data = pd.DataFrame({
            "Energy_Consumption": [energy],
            "Renewable_Percentage": [renewable],
            "GDP": [gdp]
        })

        prediction = model.predict(input_data)

        st.success(
            f"Predicted CO₂ Emissions: {prediction[0]:.2f}"
        )

except Exception as e:
    st.error(f"Error loading model: {e}")
