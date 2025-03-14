import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io
import os

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(
    filename="app.log",  # Log file
    level=logging.INFO,   # Log level (INFO, ERROR, DEBUG)
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load trained model
try:
    model = load_model("/home/user/Documents/soulai/resnet50_cifar10.h5")
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise HTTPException(status_code=500, detail="Failed to load the model.")

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Basic Authentication Setup
security = HTTPBasic()
USERNAME = "admin"
PASSWORD = "password"

# Image preprocessing function
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((32, 32))  # Resize to match model input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        return img_array, img
    except Exception as e:
        logging.error(f"Error in image preprocessing: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image file.")

# Grad-CAM Function
def generate_gradcam_heatmap(img_array, model, class_index, layer_name="conv5_block3_out"):
    grad_model = Model(
        inputs=model.input, 
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    # Ensure grads and conv_outputs are tensors
    grads = tf.convert_to_tensor(grads)  
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Ensure pooled_grads is a tensor
    pooled_grads = tf.convert_to_tensor(pooled_grads)

    # Multiply feature maps by pooled gradients
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

    return heatmap.numpy()


# Function to overlay heatmap on original image
def overlay_heatmap_on_image(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)  # Scale between 0-255
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img_cv = np.array(img)
    superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

    heatmap_path = "gradcam_output.jpg"
    cv2.imwrite(heatmap_path, superimposed_img)
    return heatmap_path

# Prediction endpoint with Basic Authentication & Grad-CAM
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    credentials: HTTPBasicCredentials = Depends(security)
):
    # Verify username and password
    if credentials.username != USERNAME or credentials.password != PASSWORD:
        logging.warning("Unauthorized access attempt.")
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    try:
        # Read image file
        image_bytes = await file.read()
        if not image_bytes:
            logging.error("Empty file received.")
            raise HTTPException(status_code=400, detail="Empty file received")

        img_array, img = preprocess_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]
        confidence = np.max(predictions)

        # Generate Grad-CAM heatmap
        heatmap = generate_gradcam_heatmap(img_array, model, predicted_class_index)
        heatmap_path = overlay_heatmap_on_image(img, heatmap)

        logging.info(f"Prediction: {predicted_class} (Confidence: {confidence:.4f}) - Heatmap saved at {heatmap_path}")

        return {
            "class": predicted_class,
            "confidence": float(confidence),
            "heatmap_path": heatmap_path
        }
    except Exception as e:
        logging.error(f"Internal Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# Run FastAPI server
if __name__ == "__main__":
    import uvicorn
    logging.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
