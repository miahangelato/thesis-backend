# ml/fingerprint_classifier.py
import os
from django.conf import settings
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - fingerprint classification disabled")
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy not available - fingerprint classification disabled")

MODEL_PATH = os.path.join(settings.BASE_DIR, "core", "improved_pattern_cnn_model.h5")
CLASS_NAMES = ["Arc", "Whorl", "Loop"]

model = None
if TENSORFLOW_AVAILABLE and NUMPY_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Fingerprint pattern model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load fingerprint model: {e}")
        model = None

def classify_fingerprint_pattern(img_file):
    if not TENSORFLOW_AVAILABLE or not NUMPY_AVAILABLE or model is None:
        return {
            'predicted_class': 'Unknown',
            'confidence': 0.0,
            'error': 'Required dependencies not available'
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
