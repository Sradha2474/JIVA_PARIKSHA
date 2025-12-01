# JIVA_PARIKSHA: AI-Powered Ayurvedic Tongue Diagnosis API

[![Deployed on Render](https://img.shields.io/badge/Render-Live-success?style=for-the-badge&logo=render)](https://jiva-pariksha.onrender.com/docs)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-009688.svg?style=for-the-badge&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.11.9-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

JIVA_PARIKSHA ("Life Examination") is a cutting-edge API that brings the ancient Ayurvedic art of tongue diagnosis (*Jivha Pariksha*) into the digital age. It uses deep learning and computer vision to analyze tongue images, determine a patient's **Prakriti/Vikriti (Dosha Imbalance)**, and generate personalized wellness recommendations.
## 🌟 KeyFeatures

* 🧠 Deep Learning Analysis: Uses a custom-trained Convolutional Neural Network (EfficientNet base) to classify tongue features into Vata, Pitta, and Kapha categories.
* 👁️ Smart Tongue Detection: Built-in computer vision algorithms (OpenCV) validate that an image actually contains a tongue before processing, checking for specific color ranges and shapes.
* ✅ Quality Control: Automatically rejects images that are too blurry, too dark, or too bright to ensure accurate diagnosis.
* 🥗 Holistic Recommendations: Returns detailed, actionable advice for Diet, Lifestyle, and Ayurvedic Treatments based on the predicted Dosha.
* 🚀 Production Ready: Built with FastAPI for high performance, utilizing lazy model loading and asynchronous request handling.

## 📡 API Endpoints
### 1. Analyze Tongue
`POST /predict`
Upload an image of a tongue to get a full diagnosis.

* **Input:** `multipart/form-data` (Key: `image`, Type: File)
* **Returns:** JSON object containing:
    * **Analysis:** Dominant dosha, confidence score, and percentage breakdown (e.g., Vata: 10%, Pitta: 85%, Kapha: 5%).
    * **Quality Metrics:** Tongue detection confidence and image quality rating.
    * **Recommendations:** Personalized diet, lifestyle tips, and suggested herbal treatments.

### 2. Health Check
`GET /health`
Verifies that the API is running and the AI models are loaded correctly.

### 3. Validation Rules
`GET /validation-info`
Returns the technical criteria used to accept or reject images (resolution, lighting thresholds, etc.).

## 🛠️ Tech Stack & Architecture
* **Framework:** FastAPI (Python)
* **ML Engine:** TensorFlow / Keras
* **Image Processing:** OpenCV (cv2) & NumPy
* **Deployment:** Render (Cloud Platform)
* **Model Architecture:**
    * **Base:** EfficientNet (Pre-trained on ImageNet)
    * **Custom Layers:** GlobalAveragePooling -> Dense -> Softmax Output
    * **Input Size:** 224x224 RGB

## 💻 Local Installation & Setup
Follow these steps to run the API on your own machine.
### Prerequisites
* Python 3.10 or 3.11
* Git
  ###Steps to run this
  1. Clone the Repository
```bash
git clone [https://github.com/Sradha2474/JIVA_PARIKSHA.git](https://github.com/Sradha2474/JIVA_PARIKSHA.git)
cd JIVA_PARIKSHA/api_deployment
2. Create Virtual Environment
Bash

python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
3. Install Dependencies
Bash

pip install -r requirements.txt
4. Run the Server
Bash

uvicorn main_api:app --reload
The API will be live at http://127.0.0.1:8000. Access the interactive documentation at http://127.0.0.1:8000/docs.

☁️ Deployment on Render
This repository is optimized for deployment on Render.com.

Connect GitHub: Create a new "Web Service" on Render and connect this repository.

Settings:

Build Command: pip install -r requirements.txt

Start Command: uvicorn main_api:app --host 0.0.0.0 --port 10000

Environment Variables:

Key: PYTHON_VERSION

Value: 3.11.9 (Crucial for TensorFlow compatibility)

📂 Project Structure
JIVA_PARIKSHA/
├── api_deployment/
│   ├── main_api.py          # The FastAPI application entry point
│   ├── requirements.txt     # Locked dependencies for production
│   └── models/              # Directory for AI assets
│       ├── ayurvedic_heuristic_model.keras  # The trained CNN
│       └── label_encoder.pkl                # Decodes predictions to text
└── README.md
Made with ❤️
