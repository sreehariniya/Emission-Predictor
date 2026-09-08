
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Sustainability Prediction",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Sustainability Prediction App")
st.write("Predict CO₂ emissions using the trained Polynomial Regression model.")

MODEL_FILE = "polynomialRegModel.pkl"
DATA_FILE = "sustainability_data.csv"

# Load model
try:
    with open(MODEL_FILE, "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load dataset
try:
    data = pd.read_csv(DATA_FILE)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

st.subheader("Enter Input Values")

# Use numeric columns from the dataset as input fields
numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()

if not numeric_columns:
    st.error("No numeric columns found in the dataset.")
    st.stop()

inputs = {}

for column in numeric_columns:
    inputs[column] = st.number_input(
        f"Enter {column}",
        value=float(data[column].mean())
    )

if st.button("Predict CO₂ Emissions 🌱"):
    input_df = pd.DataFrame([inputs])

    try:
        prediction = model.predict(input_df)
        st.success(f"Predicted CO₂ Emissions: {prediction[0]:.2f}")
    except Exception as e:
        st.error(
            "Prediction failed. Make sure the input columns match the columns "
            f"used while training the model.\n\nError: {e}"
        )

st.subheader("Dataset Preview")
st.dataframe(data.head())
