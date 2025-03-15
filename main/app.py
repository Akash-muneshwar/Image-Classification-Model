import logging
import os
import io
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Google Drive Model File
GDRIVE_FILE_ID = "11qn-bPYE-b-osdDjJy1ji7mixRkP_qSu"
MODEL_PATH = "resnet50_cifar10.h5"

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Basic Authentication Setup
security = HTTPBasic()
USERNAME = "admin"
PASSWORD = "password"

# ✅ TensorFlow Memory Optimization
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('CPU')[0], True)

# ✅ Load model at startup and keep in memory
if not os.path.exists(MODEL_PATH):
    logging.info("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)

try:
    model = load_model(MODEL_PATH)
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise RuntimeError("Failed to load the model.")

# ✅ Preprocess Image Function
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((32, 32))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img_array, img
    except Exception as e:
        logging.error(f"Error in image preprocessing: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image file.")

# ✅ Grad-CAM Optimization
def generate_gradcam_heatmap(img_array, model, class_index, layer_name="conv5_block3_out"):
    try:
        grad_model = Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])
        grad_model.trainable = False  # Prevent retraining
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_index]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)  # Normalize
        return heatmap.numpy()
    except Exception as e:
        logging.error(f"Grad-CAM error: {str(e)}")
        return None  # Return None instead of crashing

# ✅ Overlay Heatmap on Image
def overlay_heatmap_on_image(img, heatmap):
    if heatmap is None:
        return None  # No heatmap generated
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(np.array(img), 0.6, heatmap, 0.4, 0)
    heatmap_path = "gradcam_output.jpg"
    cv2.imwrite(heatmap_path, superimposed_img)
    return heatmap_path

# ✅ Home Route
@app.get("/")
def home():
    return {"message": "FastAPI is running!"}

# ✅ Prediction API
@app.post("/predict")
async def predict(file: UploadFile = File(...), credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != USERNAME or credentials.password != PASSWORD:
        logging.warning("Unauthorized access attempt.")
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    try:
        # Read image file
        image_bytes = await file.read()
        if not image_bytes:
            logging.error("Empty file received.")
            raise HTTPException(status_code=400, detail="Empty file received")

        # Preprocess image
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
            "heatmap_path": heatmap_path if heatmap_path else "Grad-CAM failed"
        }
    except Exception as e:
        logging.error(f"Internal Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# ✅ Run FastAPI Server
if __name__ == "__main__":
    logging.info("Starting FastAPI server...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
