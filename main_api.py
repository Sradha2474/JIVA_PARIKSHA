# main_api.py (Complete FastAPI Application)
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import joblib
import io
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ayurvedic_heuristic_model.keras")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
IMAGE_SIZE = (224, 224)

app = FastAPI(title="Ayurvedic Tongue Analysis API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and encoder
model = None
le = None

def load_model_and_encoder():
    """Load model and encoder with proper error handling"""
    global model, le
    try:
        logger.info("Loading encoder...")
        le = joblib.load(ENCODER_PATH)
        logger.info("Encoder loaded successfully!")
        
        logger.info("Loading model...")
        # Suppress TensorFlow warnings and set memory growth
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        # Configure TensorFlow to use CPU only to avoid GPU issues
        tf.config.set_visible_devices([], 'GPU')
        
        # Load model with custom objects if needed
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info("Model loaded successfully!")
        
        logger.info("--- Model and encoder loaded successfully. API is ready. ---")
        return True
    except Exception as e:
        logger.error(f"Error loading model or encoder: {e}")
        logger.error(f"Model path: {MODEL_PATH}")
        logger.error(f"Encoder path: {ENCODER_PATH}")
        logger.error(f"Model exists: {os.path.exists(MODEL_PATH)}")
        logger.error(f"Encoder exists: {os.path.exists(ENCODER_PATH)}")
        return False

# Don't load model on startup - load lazily when needed
logger.info("API started. Model will be loaded on first prediction request.")

@app.post("/predict")
async def handle_prediction(image: UploadFile = File(...)):
    # Load model and encoder if not already loaded
    if model is None or le is None:
        logger.info("Loading model and encoder for first prediction...")
        if not load_model_and_encoder():
            raise HTTPException(status_code=500, detail="Failed to load model or encoder. Please check server logs.")
    
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMAGE_SIZE)
        img_array = np.expand_dims(img_resized, axis=0)
        preprocessed_img = tf.keras.applications.efficientnet.preprocess_input(img_array)

        predictions = model.predict(preprocessed_img, verbose=0)[0]

        predicted_index = np.argmax(predictions)
        predicted_label = le.classes_[predicted_index]
        confidence = float(predictions[predicted_index])

        scores = {label: float(prob) for label, prob in zip(le.classes_, predictions)}

        # Generate mock recommendations based on predicted dosha
        recommendations = {
            "diet": [
                f"Eat warm, moist foods like soups and stews" if predicted_label == "vata" else
                f"Include cooling foods in your diet" if predicted_label == "pitta" else
                f"Include light, warm foods",
                f"Include healthy fats like ghee and sesame oil" if predicted_label == "vata" else
                f"Practice stress-reduction techniques" if predicted_label == "pitta" else
                f"Engage in regular physical activity"
            ],
            "treatments": [
                {
                    "name": f"Warm Oil Massage (Abhyanga)" if predicted_label == "vata" else
                           f"Cooling Therapy" if predicted_label == "pitta" else
                           f"Detox Therapy",
                    "description": f"A therapeutic massage using warm sesame oil to soothe the nervous system." if predicted_label == "vata" else
                                 f"Gentle cooling treatments to balance heat." if predicted_label == "pitta" else
                                 f"Detoxification treatments to remove toxins.",
                    "booking_link": f"https://yourwebsite.com/book/abhyanga" if predicted_label == "vata" else
                                  f"https://yourwebsite.com/book/cooling-therapy" if predicted_label == "pitta" else
                                  f"https://yourwebsite.com/book/detox-therapy"
                },
                {
                    "name": f"Steam Therapy" if predicted_label == "vata" else
                           f"Herbal Compress" if predicted_label == "pitta" else
                           f"Dry Brushing",
                    "description": f"Gentle steam therapy to relax muscles and calm nerves." if predicted_label == "vata" else
                                 f"Cooling herbal compresses to reduce inflammation." if predicted_label == "pitta" else
                                 f"Dry brushing to stimulate circulation and lymphatic drainage.",
                    "booking_link": f"https://yourwebsite.com/book/steam-therapy" if predicted_label == "vata" else
                                  f"https://yourwebsite.com/book/herbal-compress" if predicted_label == "pitta" else
                                  f"https://yourwebsite.com/book/dry-brushing"
                }
            ],
            "exercises": [
                f"Restorative yoga poses like child's pose and corpse pose" if predicted_label == "vata" else
                f"Moderate intensity yoga and swimming" if predicted_label == "pitta" else
                f"Vigorous exercise and dynamic yoga",
                f"Gentle stretching exercises" if predicted_label == "vata" else
                f"Cooling breathing exercises" if predicted_label == "pitta" else
                f"Strength training and cardio"
            ],
            "tips": [
                f"Keep yourself warm, especially in cold weather" if predicted_label == "vata" else
                f"Stay cool and avoid overheating" if predicted_label == "pitta" else
                f"Maintain regular exercise routine",
                f"Avoid dry and cold environments" if predicted_label == "vata" else
                f"Avoid spicy and hot foods" if predicted_label == "pitta" else
                f"Avoid heavy, oily foods"
            ]
        }

        return {
            "dominant_dosha": predicted_label,
            "confidence": confidence,
            "all_scores": scores,
            "recommendations": recommendations
        }
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or le is None:
        return {"status": "unhealthy", "message": "Model or encoder not loaded"}
    return {"status": "healthy", "message": "API is ready"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)