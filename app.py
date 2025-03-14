import streamlit as st
import requests
import os
from PIL import Image

# FastAPI Backend URL (Update with your actual running URL)
FASTAPI_URL = "http://127.0.0.1:8000/predict"  # Change if deploying on Render

# Set Page Title
st.title("Image Classification with Grad-CAM")

# File Uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image temporarily
    image_path = "temp_image.jpg"
    image.save(image_path)

    # Send the image to FastAPI for prediction
    with st.spinner("Classifying image..."):
        with open(image_path, "rb") as img_file:
            response = requests.post(
                FASTAPI_URL,
                files={"file": img_file},
                auth=("admin", "password")  # FastAPI Authentication
            )

        if response.status_code == 200:
            result = response.json()
            st.success(f"**Predicted Class:** {result['class']} (Confidence: {result['confidence']:.4f})")

            # Display Grad-CAM heatmap
            if "heatmap_path" in result:
                heatmap_path = result["heatmap_path"]
                if os.path.exists(heatmap_path):
                    st.image(heatmap_path, caption="Grad-CAM Heatmap", use_column_width=True)
                else:
                    st.warning("Heatmap not found. Please check the server.")
        else:
            st.error("Error in prediction. Please try again.")

    # Remove temporary image file
    os.remove(image_path)
