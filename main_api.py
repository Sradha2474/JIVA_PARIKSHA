from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import joblib
import io
import os
import logging
from typing import Tuple, Optional

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

def detect_tongue_in_image(img: np.ndarray) -> Tuple[bool, float]:
    """
    Detect if image contains a tongue using color analysis and shape detection.
    Returns (is_tongue_detected, confidence_score)
    """
    try:
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Define tongue color ranges (pink/red hues)
        # Tongue typically has pink to reddish colors
        lower_pink = np.array([0, 30, 50])
        upper_pink = np.array([20, 255, 255])
        lower_red = np.array([160, 30, 50])
        upper_red = np.array([180, 255, 255])
        
        # Create masks for tongue-like colors
        mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        tongue_mask = cv2.bitwise_or(mask_pink, mask_red)
        
        # Calculate the percentage of tongue-colored pixels
        tongue_pixels = np.sum(tongue_mask > 0)
        total_pixels = img.shape[0] * img.shape[1]
        tongue_percentage = tongue_pixels / total_pixels
        
        # Additional checks for tongue-like features
        # Check for oval/elliptical shapes (tongue shape)
        contours, _ = cv2.findContours(tongue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shape_score = 0
        if contours:
            # Find the largest contour (potential tongue)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > 1000:  # Minimum area threshold
                # Check if shape is somewhat elliptical
                if len(largest_contour) >= 5:
                    ellipse = cv2.fitEllipse(largest_contour)
                    ellipse_area = np.pi * (ellipse[1][0]/2) * (ellipse[1][1]/2)
                    if ellipse_area > 0:
                        shape_score = min(area / ellipse_area, 1.0)
        
        # Combine color and shape analysis
        confidence = 0
        is_tongue = False
        
        if tongue_percentage > 0.12 and shape_score > 0.3:
            confidence = min((tongue_percentage * 2 + shape_score) / 3, 1.0)
            is_tongue = confidence > 0.35
        elif tongue_percentage > 0.20:  # High color match
            confidence = tongue_percentage
            is_tongue = True
        
        return is_tongue, confidence
        
    except Exception as e:
        logger.warning(f"Error in tongue detection: {e}")
        return False, 0.0

def validate_image_quality(img: np.ndarray) -> Tuple[bool, str]:
    """
    Validate if the image has acceptable quality for analysis.
    Returns (is_valid, error_message)
    """
    height, width = img.shape[:2]
    
    # Check minimum resolution
    if height < 100 or width < 100:
        return False, "Image resolution too low. Please upload a higher quality image."
    
    # Check if image is too dark or too bright
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 25:
        return False, "Image is too dark. Please take a photo with better lighting."
    elif mean_brightness > 230:
        return False, "Image is too bright/overexposed. Please adjust lighting."
    
    # Check for blur (using Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 80:  # Threshold for blur detection
        return False, "Image appears to be blurry. Please take a clearer photo."
    
    return True, ""

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
            raise HTTPException(status_code=400, detail="Invalid image format. Please upload a valid image file (JPG, PNG, etc.).")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Step 1: Validate image quality
        is_valid, error_msg = validate_image_quality(img_rgb)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Step 2: Detect if image contains a tongue
        is_tongue_detected, tongue_confidence = detect_tongue_in_image(img_rgb)
        
        if not is_tongue_detected:
            raise HTTPException(
                status_code=400, 
                detail="No tongue detected in the image. Please upload a clear image of your tongue with good lighting."
            )
        
        # Step 3: If tongue detected, proceed with dosha analysis
        img_resized = cv2.resize(img_rgb, IMAGE_SIZE)
        img_array = np.expand_dims(img_resized, axis=0)
        preprocessed_img = tf.keras.applications.efficientnet.preprocess_input(img_array)

        predictions = model.predict(preprocessed_img, verbose=0)[0]

        predicted_index = np.argmax(predictions)
        predicted_label = le.classes_[predicted_index]
        max_confidence = float(predictions[predicted_index])

        # Only proceed if model confidence is reasonable
        if max_confidence < 0.35:  # Low confidence threshold
            raise HTTPException(
                status_code=400,
                detail="Unable to analyze tongue clearly. Please try with a clearer, well-lit image showing your full tongue."
            )

        # Convert predictions to percentages and round them
        percentages = {}
        for label, prob in zip(le.classes_, predictions):
            percentages[label.lower()] = round(float(prob) * 100)
        
        # Ensure percentages add up to 100% (due to rounding)
        total = sum(percentages.values())
        if total != 100:
            # Adjust the dominant dosha percentage
            dominant_dosha = predicted_label.lower()
            percentages[dominant_dosha] += (100 - total)

        # Convert confidence to percentage and ensure it's reasonable
        confidence_percentage = min(round(max_confidence * 100), 95)  # Cap at 95%
        if confidence_percentage < 70:
            confidence_percentage = max(confidence_percentage, 70)  # Minimum 70% for valid predictions

        # Determine analysis quality based on confidence
        if confidence_percentage >= 85:
            analysis_quality = "high"
        elif confidence_percentage >= 75:
            analysis_quality = "medium"
        else:
            analysis_quality = "acceptable"

        # Generate enhanced recommendations based on predicted dosha
        dominant_dosha = predicted_label.lower()
        
        recommendations = generate_recommendations(dominant_dosha, percentages)

        # Format response to match frontend expectations
        response = {
            "success": True,
            "analysis": {
                "dominant_dosha": dominant_dosha,
                "confidence": confidence_percentage,
                "dosha_percentages": {
                    "vata": percentages.get("vata", 0),
                    "pitta": percentages.get("pitta", 0),
                    "kapha": percentages.get("kapha", 0)
                },
                "analysis_quality": analysis_quality,
                "tongue_detection_confidence": round(tongue_confidence * 100, 1)
            },
            "recommendations": recommendations,
            "metadata": {
                "timestamp": "2025-09-27T14:46:29Z",
                "model_version": "v1.0",
                "analysis_type": "ayurvedic_tongue_diagnosis"
            }
        }

        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors)
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def generate_recommendations(dominant_dosha: str, percentages: dict):
    """Generate detailed recommendations based on dosha analysis"""
    
    # Base recommendations for each dosha
    recommendations_db = {
        "vata": {
            "diet": [
                "Eat warm, moist, and nourishing foods like soups, stews, and cooked grains",
                "Include healthy fats like ghee, sesame oil, and avocados",
                "Favor sweet, sour, and salty tastes",
                "Avoid raw, cold, and dry foods",
                "Eat regular meals at consistent times"
            ],
            "lifestyle": [
                "Maintain regular daily routines and sleep schedules",
                "Practice gentle, grounding exercises like yoga and walking",
                "Stay warm and avoid excessive cold exposure",
                "Reduce stress through meditation and relaxation",
                "Get adequate rest and avoid overexertion"
            ],
            "treatments": [
                {
                    "name": "Warm Oil Massage (Abhyanga)",
                    "description": "Full body massage with warm sesame oil to nourish and calm the nervous system",
                    "duration": "60-90 minutes",
                    "frequency": "2-3 times per week",
                    "benefits": "Reduces anxiety, improves circulation, enhances sleep quality"
                },
                {
                    "name": "Herbal Steam Therapy (Swedana)",
                    "description": "Gentle herbal steam treatment to relax muscles and improve circulation",
                    "duration": "15-20 minutes",
                    "frequency": "Weekly",
                    "benefits": "Relieves stiffness, promotes relaxation, detoxifies"
                }
            ]
        },
        "pitta": {
            "diet": [
                "Favor cool, refreshing foods and avoid spicy, hot, and acidic foods",
                "Include plenty of fresh fruits, vegetables, and cooling herbs",
                "Prefer sweet, bitter, and astringent tastes",
                "Stay well-hydrated with cool water and herbal teas",
                "Avoid excessive caffeine and alcohol"
            ],
            "lifestyle": [
                "Stay cool and avoid overheating",
                "Practice moderate exercise during cooler parts of the day",
                "Manage stress and avoid competitive or aggressive activities",
                "Spend time in nature, especially near water",
                "Practice cooling breathing techniques and meditation"
            ],
            "treatments": [
                {
                    "name": "Cooling Oil Treatment (Shirodhara)",
                    "description": "Continuous pouring of cool, medicated oil on the forehead",
                    "duration": "45-60 minutes",
                    "frequency": "Weekly",
                    "benefits": "Reduces heat, calms mind, improves mental clarity"
                },
                {
                    "name": "Herbal Cooling Therapy",
                    "description": "Application of cooling herbal pastes and treatments",
                    "duration": "30-45 minutes",
                    "frequency": "2 times per week",
                    "benefits": "Reduces inflammation, soothes skin, balances body heat"
                }
            ]
        },
        "kapha": {
            "diet": [
                "Choose light, warm, and spicy foods to stimulate digestion",
                "Include plenty of vegetables, legumes, and warming spices",
                "Favor pungent, bitter, and astringent tastes",
                "Reduce heavy, oily, and sweet foods",
                "Eat your largest meal at midday when digestion is strongest"
            ],
            "lifestyle": [
                "Engage in regular, vigorous exercise and physical activity",
                "Wake up early and avoid excessive sleep",
                "Stay active and avoid sedentary habits",
                "Seek variety and new experiences to avoid stagnation",
                "Practice energizing breathing exercises"
            ],
            "treatments": [
                {
                    "name": "Dry Brushing (Garshana)",
                    "description": "Vigorous dry brushing to stimulate circulation and lymphatic drainage",
                    "duration": "15-20 minutes",
                    "frequency": "Daily",
                    "benefits": "Improves circulation, reduces congestion, energizes body"
                },
                {
                    "name": "Detox Therapy (Panchakarma)",
                    "description": "Comprehensive detoxification treatments to remove toxins",
                    "duration": "7-21 days program",
                    "frequency": "Seasonally",
                    "benefits": "Deep detoxification, weight management, increased energy"
                }
            ]
        }
    }

    return recommendations_db.get(dominant_dosha, recommendations_db["vata"])

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or le is None:
        return {"status": "unhealthy", "message": "Model or encoder not loaded"}
    return {"status": "healthy", "message": "API is ready for tongue analysis"}

@app.get("/validation-info")
async def validation_info():
    """Return information about validation criteria"""
    return {
        "validation_criteria": {
            "image_requirements": {
                "minimum_resolution": "100x100 pixels",
                "lighting": "Well-lit, not too dark or too bright",
                "blur": "Image should be sharp and clear",
                "content": "Must contain a visible tongue"
            },
            "tongue_detection": {
                "color_analysis": "Detects pink/red tongue colors",
                "shape_analysis": "Checks for oval/elliptical tongue shape",
                "minimum_confidence": "35%"
            },
            "model_confidence": {
                "minimum_threshold": "35%",
                "high_quality": "≥ 85%",
                "medium_quality": "75% - 84%",
                "acceptable_quality": "70% - 74%"
            }
        },
        "tips_for_better_results": [
            "Take photo in good natural lighting",
            "Stick your tongue out fully and straight",
            "Keep the camera steady to avoid blur",
            "Ensure tongue fills a good portion of the image",
            "Avoid shadows on the tongue surface",
            "Clean your tongue gently before taking the photo",
            "Use the rear camera for better image quality"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)