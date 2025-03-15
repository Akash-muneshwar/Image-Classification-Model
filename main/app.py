import streamlit as st
import requests
from PIL import Image
import io
#
#https://image-classification-model-5mpp.onrender.com/
# FastAPI backend URL (update this if needed)
API_URL = "https://image-classification-model-5mpp.onrender.com/predict"
USERNAME = "admin"
PASSWORD = "password"

st.title("🖼️ Image Classification Web App")
st.write("Upload an image to classify it using the trained model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")
    
    # Convert image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image.format)
    image_bytes = image_bytes.getvalue()

    # Debugging prints
    print("Sending request to:", API_URL)  
    print("File selected:", uploaded_file.name)  
    print("File type:", uploaded_file.type)  

    # Send request to FastAPI
    """response = requests.post(
        API_URL,
        auth=(USERNAME, PASSWORD),
        files={"file": (uploaded_file.name, image_bytes)}

    )"""
    response = requests.post(
    API_URL,
    auth=(USERNAME, PASSWORD),
    files={"file": ("image.jpg", image_bytes, "image/jpeg")}
)


    # Debugging response
    print("Response status code:", response.status_code)  
    print("Response body:", response.text)  # Print full response from API


    # Send request to FastAPI
    """response = requests.post(
        API_URL,
        auth=(USERNAME, PASSWORD),
        files={"file": (uploaded_file.name, image_bytes, uploaded_file.type)}
    )"""

    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: **{result['class']}**")
        st.write(f"Confidence: **{result['confidence']*100:.2f}%**")
        st.image(result['heatmap_path'], caption="Grad-CAM Heatmap", use_column_width=True)
    else:
        st.error("Failed to get prediction. Check the backend logs.")
