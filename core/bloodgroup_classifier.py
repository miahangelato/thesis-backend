import os
import logging

# Conditional imports for ML libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    print("NumPy not available - blood group classification disabled")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    print("OpenCV not available - blood group classification disabled")

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - blood group classification disabled")

logger = logging.getLogger(__name__)

class BloodGroupClassifier:
    def __init__(self):
        self.model = None
        self.model_path = os.path.join(os.path.dirname(__file__), 'bloodgroup_model_20250823-140933.h5')
        self.load_model()
        
        # Blood group classes based on the Kaggle dataset
        self.blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
    
    def load_model(self):
        """Load the blood group classification model"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available - blood group classification disabled")
            return
            
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                logger.info(f"Blood group model loaded successfully from {self.model_path}")
            else:
                logger.error(f"Model file not found at {self.model_path}")
                logger.info("Blood group classification will not be available")
        except Exception as e:
            logger.error(f"Failed to load blood group model: {e}")
            self.model = None

    def preprocess_fingerprint(self, image_path):
        """
        Preprocess fingerprint image for blood group prediction
        
        Args:
            image_path (str): Path to the fingerprint image
            
        Returns:
            numpy.ndarray: Preprocessed image ready for prediction
        """
        if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
            logger.warning("OpenCV and NumPy are required for image preprocessing but not available")
            return None
            
        print(f"[DEBUG] Processing image: {image_path}")
        
        try:
            # Read the image
            img = cv2.imread(image_path)
            print(f"[DEBUG] After cv2.imread: shape={None if img is None else img.shape}, dtype={None if img is None else img.dtype}")
            if img is None:
                raise ValueError(f"Could not read image from {image_path}")

            # Resize to (128, 128)
            img = cv2.resize(img, (128, 128))
            print(f"[DEBUG] After cv2.resize: shape={img.shape}, dtype={img.dtype}")

            # Normalize pixel values
            img = img.astype(np.float32) / 255.0
            print(f"[DEBUG] After normalization: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")

            # Ensure the image has 3 channels (convert grayscale to RGB)
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 3:  # Already RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"[DEBUG] After color conversion: shape={img.shape}, dtype={img.dtype}")

            # Expand dims to (1, 128, 128, 3)
            img = np.expand_dims(img, axis=0)
            print(f"[DEBUG] After np.expand_dims: shape={img.shape}, dtype={img.dtype}")

            return img
        except Exception as e:
            print(f"[ERROR] Error in preprocess_fingerprint: {e}")
            raise

    def predict_blood_group(self, fingerprint_image_path):
        """
        Predict blood group from fingerprint image
        
        Args:
            fingerprint_image_path (str): Path to the fingerprint image
            
        Returns:
            dict: {
                'predicted_blood_group': str,
                'confidence': float,
                'all_probabilities': dict
            }
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return {
                'predicted_blood_group': 'Unknown',
                'confidence': 0.0,
                'all_probabilities': {bg: 0.125 for bg in self.blood_groups},
                'error': 'TensorFlow model not available'
            }
            
        if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
            return {
                'predicted_blood_group': 'Unknown',
                'confidence': 0.0,
                'all_probabilities': {bg: 0.125 for bg in self.blood_groups},
                'error': 'OpenCV and NumPy required but not available'
            }
            
        try:
            # Preprocess the image
            processed_image = self.preprocess_fingerprint(fingerprint_image_path)
            if processed_image is None:
                return {
                    'predicted_blood_group': 'Error',
                    'confidence': 0.0,
                    'all_probabilities': {},
                    'error': 'Failed to preprocess image'
                }
                
            print(f"[DEBUG] Model input shape: {processed_image.shape}, dtype={processed_image.dtype}")
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            print(f"[DEBUG] Model output shape: {predictions.shape}, dtype={predictions.dtype}")
            
            # Get the predicted class and confidence
            predicted_class_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_index])
            predicted_blood_group = self.blood_groups[predicted_class_index]
            
            # Create probability dictionary for all classes
            all_probabilities = {}
            for i, blood_group in enumerate(self.blood_groups):
                all_probabilities[blood_group] = float(predictions[0][i])
            
            result = {
                'predicted_blood_group': predicted_blood_group,
                'confidence': confidence,
                'all_probabilities': all_probabilities
            }
            
            print(f"[DEBUG] Final result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error during blood group prediction: {e}")
            return {
                'predicted_blood_group': 'Error',
                'confidence': 0.0,
                'all_probabilities': {},
                'error': str(e)
            }

    def predict_blood_group_batch(self, fingerprint_paths):
        """
        Predict blood group from multiple fingerprint images and return the most confident prediction
        
        Args:
            fingerprint_paths (list): List of paths to fingerprint images
            
        Returns:
            dict: Best prediction result
        """
        if not fingerprint_paths:
            return {
                'predicted_blood_group': 'Error',
                'confidence': 0.0,
                'all_probabilities': {},
                'error': 'No fingerprint images provided'
            }
        
        best_prediction = None
        best_confidence = 0.0
        
        for path in fingerprint_paths:
            try:
                prediction = self.predict_blood_group(path)
                if prediction.get('confidence', 0) > best_confidence:
                    best_confidence = prediction.get('confidence', 0)
                    best_prediction = prediction
            except Exception as e:
                logger.error(f"Error predicting blood group for {path}: {e}")
                continue
        
        if best_prediction is None:
            return {
                'predicted_blood_group': 'Error',
                'confidence': 0.0,
                'all_probabilities': {},
                'error': 'Failed to predict from any image'
            }
        
        return best_prediction

# Global instance
try:
    blood_group_classifier = BloodGroupClassifier()
    print("INFO Blood group model loaded successfully")
except Exception as e:
    print(f"ERROR Failed to initialize blood group classifier: {e}")
    blood_group_classifier = None

# Convenience functions for external use
def classify_blood_group_from_multiple(fingerprint_paths):
    """
    Classify blood group from multiple fingerprint images
    
    Args:
        fingerprint_paths (list): List of paths to fingerprint images
        
    Returns:
        dict: Blood group prediction result
    """
    if blood_group_classifier is None:
        return {
            'predicted_blood_group': 'Error',
            'confidence': 0.0,
            'all_probabilities': {},
            'error': 'Blood group classifier not initialized'
        }
    
    return blood_group_classifier.predict_blood_group_batch(fingerprint_paths)

def classify_blood_group_single(fingerprint_path):
    """
    Classify blood group from a single fingerprint image
    
    Args:
        fingerprint_path (str): Path to fingerprint image
        
    Returns:
        dict: Blood group prediction result
    """
    if blood_group_classifier is None:
        return {
            'predicted_blood_group': 'Error',
            'confidence': 0.0,
            'all_probabilities': {},
            'error': 'Blood group classifier not initialized'
        }
    
    return blood_group_classifier.predict_blood_group(fingerprint_path)
