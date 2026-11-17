
import streamlit as st
import numpy as np
import pandas as pd
import joblib


# Page Configuration
st.set_page_config(
    page_title="Wine Clustering",
    page_icon="🍷",
    layout="wide"
)


# Load saved models
model = joblib.load("wine_kmeans.pkl")
scaler = joblib.load("wine_scaler.pkl")
pca = joblib.load("wine_pca.pkl")


# Title
st.title("🍷 Wine Grouping Using K-Means")
st.write("Group wines based on their chemical properties.")

st.divider()


# Wine Features
features = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline"
]


# Input Section
st.subheader("Enter Wine Details")

col1, col2 = st.columns(2)

input_values = []

# Default values for the 13 features
defaults = [
    13.0, 2.0, 2.3, 17.0, 100.0,
    2.5, 2.0, 0.3, 1.5, 5.0,
    1.0, 3.0, 1000.0
]

for i, feature in enumerate(features):

    if i < 7:
        with col1:
            value = st.number_input(
                feature.replace("_", " ").title(),
                value=float(defaults[i])
            )
    else:
        with col2:
            value = st.number_input(
                feature.replace("_", " ").title(),
                value=float(defaults[i])
            )

    input_values.append(value)


st.divider()


# Prediction
if st.button(
    "Predict Wine Cluster",
    type="primary",
    use_container_width=True
):

    # Prepare input data
    input_data = pd.DataFrame(
        [input_values],
        columns=features
    )

    # Scale input data
    scaled_data = scaler.transform(input_data)

    # Predict cluster
    prediction = model.predict(scaled_data)[0]

    # Display result
    st.success("Clustering Completed!")

    st.subheader("Prediction Result")

    st.metric(
        label="Predicted Wine Cluster",
        value=f"Cluster {prediction}"
    )

    # Display PCA information
    pca_data = pca.transform(scaled_data)

    st.write("### PCA Transformed Values")

    st.write(pca_data)

    st.info(
        f"This wine belongs to Cluster {prediction} "
        "based on its chemical properties."
    )


st.divider()

st.caption(
    "Wine Grouping System | Unsupervised Machine Learning | K-Means"
)
