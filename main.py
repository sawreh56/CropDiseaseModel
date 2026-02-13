from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import io

# -----------------------------
# Create FastAPI App
# -----------------------------
app = FastAPI()

# -----------------------------
# Load Model on Startup
# -----------------------------
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = tf.keras.models.load_model("crop_disease_model.keras")
        print("Model loaded successfully")
    except Exception as e:
        print("Model loading failed:", e)

# -----------------------------
# Load Class Names & Disease Info
# -----------------------------
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

with open("disease_info.pkl", "rb") as f:
    disease_info = pickle.load(f)

# -----------------------------
# Home Route
# -----------------------------
@app.get("/")
def home():
    return {"message": "🌿 Crop Disease Detection API is running successfully!"}

# -----------------------------
# Prediction Route
# -----------------------------
@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):

    # Safety check if model failed to load
    if model is None:
        return {"error": "Model not loaded"}

    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Resize
    image = image.resize((224, 224))

    # Convert to array
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)

    predicted_class = class_names[predicted_index]

    # Get extra info
    precautions = disease_info[predicted_class]["precautions"]
    medicine = disease_info[predicted_class]["medicine"]

    return {
        "disease": predicted_class,
        "confidence": round(confidence, 2),
        "precautions": precautions,
        "medicine": medicine
    }
