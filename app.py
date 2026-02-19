# ==========================================
# Hierarchical Clustering - Iris Segmentation
# ==========================================

import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Iris Clustering App",
    page_icon="🌸",
    layout="wide"
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
    <style>
    h1 {
        color: #4a148c;
        text-align: center;
    }
    .stButton>button {
        background-color: #4a148c;
        color: white;
        height: 50px;
        width: 100%;
        border-radius: 8px;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌸 Iris Flower Segmentation")
st.write("Hierarchical Clustering using Agglomerative Method")

# -----------------------------
# Load Model & Scaler
# -----------------------------
model = joblib.load("hierarchical_iris_model.pkl")
scaler = joblib.load("iris_scaler.pkl")

# -----------------------------
# Input Section
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", value=5.0)
    sepal_width = st.number_input("Sepal Width (cm)", value=3.5)

with col2:
    petal_length = st.number_input("Petal Length (cm)", value=1.4)
    petal_width = st.number_input("Petal Width (cm)", value=0.2)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Cluster"):

    input_data = np.array([[sepal_length,
                            sepal_width,
                            petal_length,
                            petal_width]])

    scaled_input = scaler.transform(input_data)

    cluster = model.fit_predict(scaled_input)[0]

    st.subheader("Cluster Assignment:")
    st.success(f"Cluster {cluster}")

    # Interpretation
    if cluster == 0:
        st.info("Likely resembles Setosa-type cluster characteristics")
    elif cluster == 1:
        st.warning("Likely resembles Versicolor-type cluster characteristics")
    else:
        st.error("Likely resembles Virginica-type cluster characteristics")