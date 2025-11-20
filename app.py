import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle

# ---------------------------
# Load Model & Scaler
# ---------------------------

st.title("📡 Signal Strength Classification App")
st.write("Predict signal quality (classes 3–8) using a trained Neural Network.")

# Load Keras model
model = load_model("final_model.keras")

# Load scaler
scaler = StandardScaler()

# Mapping back to original labels
label_reverse_map = {0:3, 1:4, 2:5, 3:6, 4:7, 5:8}

# ---------------------------
# Sample CSV Download Section
# ---------------------------

st.info("Download the sample **test.csv** file, fill it with your own data, and upload it below.")

with open("test.csv", "rb") as f:
    st.download_button(
        label="⬇️ Download Sample CSV",
        data=f,
        file_name="test.csv",
        mime="text/csv"
    )

# ---------------------------
# Session State Initialization
# ---------------------------

if "data" not in st.session_state:
    st.session_state.data = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploaded_file = st.file_uploader(
    "📂 Upload Signal Parameters CSV",
    type=["csv"],
    key=f"uploader_{st.session_state.uploader_key}"
)

# ---------------------------
# Read Uploaded File
# ---------------------------

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.data = df
    st.success("✅ File uploaded successfully!")
    st.write("### 📄 Data Preview")
    st.write(df.head())

# ---------------------------
# Buttons
# ---------------------------

col1, col2 = st.columns(2)
with col1:
    predict_btn = st.button("🚀 Predict Signal Strength")
with col2:
    clear_btn = st.button("🔄 Reset")

# Reset App
if clear_btn:
    st.session_state.data = None
    st.session_state.uploader_key += 1
    st.rerun()

# ---------------------------
# Prediction Logic
# ---------------------------

if predict_btn:

    if st.session_state.data is None:
        st.warning("⚠️ Please upload a CSV file first.")
    else:
        data = st.session_state.data.copy()

        # Clean Uploaded Data
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)

        # Scale Input Features
        scaled_data = scaler.fit_transform(data)

        # Predict
        predictions = model.predict(scaled_data)
        predicted_labels_numeric = np.argmax(predictions, axis=1)

        # Convert to original: 0→3, 1→4, ... 5→8
        predicted_signal_strength = [label_reverse_map[i] for i in predicted_labels_numeric]

        # Prepare Results
        results = data.copy()
        results["Predicted_Signal_Strength"] = predicted_signal_strength

        st.write("### 📈 Prediction Results")
        st.write(results.head())

        # Download
        st.download_button(
            label="⬇️ Download Predictions",
            data=results.to_csv(index=False),
            file_name="signal_strength_predictions.csv",
            mime="text/csv"
        )
