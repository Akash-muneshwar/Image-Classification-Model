# Image Classification API with FastAPI & Streamlit  

This project is an image classification model using ResNet50 trained on CIFAR-10. The backend is built with FastAPI, and the frontend is powered by Streamlit. The model predicts the class of an image and generates a Grad-CAM heatmap for interpretability.  

## Setup & Installation  

### 1. Clone the Repository  
git clone https://github.com/your-username/image-classification.git  
cd image-classification  

### 2. Install Dependencies  
pip install -r requirements.txt  

### 3. Run FastAPI Backend  
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000  

### 4. Run Streamlit Frontend  
streamlit run app.py  

## How to Use  

1. Upload an image through the Streamlit UI.  
2. The FastAPI backend processes the image and returns a prediction.  
3. The predicted class and a Grad-CAM heatmap will be displayed.  

## API Endpoints  

- **GET /** → Check if the API is running.  
- **POST /predict** → Upload an image to get predictions.  

Example `POST` request using cURL:  
curl -X 'POST' "http://localhost:8000/predict" \  
  -H "accept: application/json" \  
  -H "Authorization: Basic $(echo -n 'admin:password' | base64)" \  
  -H "Content-Type: multipart/form-data" \  
  -F "file=@image.jpg"  

🔹 **Note:** Replace `localhost:8000` with the deployed API URL if running on Render.  

## Deployment  

The backend was deployed on [Render](https://render.com/). Steps:  
1. Linked the GitHub repository to Render.  
2. Deployed the FastAPI backend as a web service.  
3. Updated the Streamlit frontend to use the deployed API.  

## Project Summary  

- Used ResNet50 for image classification.  
- Built a FastAPI backend for predictions.  
- Implemented Grad-CAM for visualization.  
- Developed a Streamlit UI for user interaction.    
