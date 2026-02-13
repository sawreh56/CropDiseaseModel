from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import io

app = FastAPI()

model = None
class_names = None
disease_info = None

@app.on_event("startup")
def load_model():
    global model, class_names, disease_info
    try:
        model = tf.keras.models.load_model("crop_disease_model.keras")
        with open("class_names.pkl", "rb") as f:
            class_names = pickle.load(f)
        with open("disease_info.pkl", "rb") as f:
            disease_info = pickle.load(f)
        print("Model and data loaded successfully")
    except Exception as e:
        print(f"Error loading model or data: {e}")

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    global model, class_names, disease_info
    if model is None:
        return {"error": "Model not loaded"}

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)
    predicted_class = class_names[predicted_index]
    precautions = disease_info[predicted_class]["precautions"]
    medicine = disease_info[predicted_class]["medicine"]

    return {
        "disease": predicted_class,
        "confidence": round(confidence, 2),
        "precautions": precautions,
        "medicine": medicine
    }
