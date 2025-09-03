# ml/fingerprint_classifier.py
import os
import platform
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

# Platform check
IS_PRODUCTION = os.environ.get('RAILWAY_DEPLOYMENT') == 'True'
IS_WINDOWS = platform.system() == 'Windows'

# Conditional imports for ML libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    logger.warning("NumPy not available - fingerprint classification will be limited")

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    logger.warning("TensorFlow not available - fingerprint classification will be limited")

# S3 Model path for production
S3_MODEL_KEY = 'models/improved_pattern_cnn_model.h5'

# Local model path for development
LOCAL_MODEL_PATH = os.path.join(settings.BASE_DIR, "core", "improved_pattern_cnn_model.h5")

# List of supported pattern classes
CLASS_NAMES = ["Arc", "Whorl", "Loop"]

# Global model variable
model = None

def load_model():
    """Load the model from either S3 or local file system"""
    global model
    
    if not TENSORFLOW_AVAILABLE or not NUMPY_AVAILABLE:
        logger.warning("TensorFlow or NumPy not available. Cannot load fingerprint model.")
        return None
        
    # Check if we should use S3
    if IS_PRODUCTION:
        try:
            from .s3_model_loader import load_model_from_s3
            logger.info(f"Loading fingerprint model from S3: {S3_MODEL_KEY}")
            model = load_model_from_s3(S3_MODEL_KEY)
            logger.info("Fingerprint pattern model loaded from S3")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from S3: {e}")
            # Fall back to local file if available
    
    # Use local file
    try:
        if os.path.exists(LOCAL_MODEL_PATH):
            logger.info(f"Loading fingerprint model from local path: {LOCAL_MODEL_PATH}")
            model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
            logger.info("Fingerprint pattern model loaded from local file")
            return model
        else:
            logger.error(f"Model not found at {LOCAL_MODEL_PATH}")
            return None
    except Exception as e:
        logger.error(f"Failed to load fingerprint model: {e}")
        return None

def classify_fingerprint_pattern(img_file):
    global model
    
    # Load model if not already loaded
    if model is None:
        model = load_model()
        
    if not TENSORFLOW_AVAILABLE or not NUMPY_AVAILABLE or model is None:
        return {
            'predicted_class': 'Unknown',
            'confidence': 0.0,
            'error': 'Required dependencies or model not available'
        }
        
        try:
            img = image.load_img(img_file, color_mode="grayscale", target_size=(128, 128))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) / 255.0
            preds = model.predict(x)
            predicted_class = CLASS_NAMES[np.argmax(preds)]
            confidence = float(np.max(preds))
            return {
                'predicted_class': predicted_class,
                'confidence': confidence
            }
        except Exception as e:
            return {
                'predicted_class': 'Error',
                'confidence': 0.0,
                'error': str(e)
            }
