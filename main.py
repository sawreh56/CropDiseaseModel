from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import io

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="🌿 Crop Disease Detection API")

# -----------------------------
# Global Model Variable
# -----------------------------
model = None
class_names = []
disease_info = {}

# -----------------------------
# Startup Event: Load Model & Pickle Files
# -----------------------------
@app.on_event("startup")
def load_resources():
    global model, class_names, disease_info
    try:
        model = tf.keras.models.load_model("crop_disease_model.keras")
        print("✅ Model loaded successfully")
    except Exception as e:
        print("❌ Model loading failed:", e)

    try:
        with open("class_names.pkl", "rb") as f:
            class_names = pickle.load(f)
        with open("disease_info.pkl", "rb") as f:
            disease_info = pickle.load(f)
        print("✅ Pickle files loaded successfully")
    except Exception as e:
        print("❌ Pickle loading failed:", e)

# -----------------------------
# Home Route
# -----------------------------
@app.get("/")
def home():
    return {"message": "🌿 Crop Disease Detection API is running successfully!"}

# -----------------------------
# Predict Route
# -----------------------------
@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):

    if not model:
        return {"error": "Model not loaded yet."}

    # Read Image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Resize & Normalize
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)
    predicted_class = class_names[predicted_index]

    # Extra Info
    precautions = disease_info[predicted_class]["precautions"]
    medicine = disease_info[predicted_class]["medicine"]

    return {
        "disease": predicted_class,
        "confidence": round(confidence, 2),
        "precautions": precautions,
        "medicine": medicine
    }
