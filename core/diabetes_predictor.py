

import pandas as pd
import numpy as np
import pickle
import os
import logging
from django.conf import settings
from .models import Participant, Fingerprint
from .risk_thresholds import get_risk_category, get_risk_description

logger = logging.getLogger(__name__)

# Check if we're in production environment
IS_PRODUCTION = os.environ.get('RAILWAY_DEPLOYMENT') == 'True'

# S3 keys for model files
S3_MODEL_KEYS = {
    'A': 'models/diabetes_risk_model.pkl',
    'B': 'models/diabetes_risk_model_B.pkl',
}

S3_COLS_KEYS = {
    'A': 'models/diabetes_risk_model_columns.pkl',
    'B': 'models/diabetes_risk_model_columns_B.pkl',
}

class DiabetesPredictor:
    def __init__(self):
        base = settings.BASE_DIR
        # Local paths
        self.model_paths = {
            'A': os.path.join(base, "core", "diabetes_risk_model.pkl"),
            'B': os.path.join(base, "core", "diabetes_risk_model_B.pkl"),
        }
        self.cols_paths = {
            'A': os.path.join(base, "core", "diabetes_risk_model_columns.pkl"),
            'B': os.path.join(base, "core", "diabetes_risk_model_columns_B.pkl"),
        }
        self.models = {}
        self.model_columns = {}
        # We'll lazy load the models when needed

    def load_models(self):
        """Load ML models either from S3 (production) or local file system (development)"""
        if IS_PRODUCTION:
            try:
                from .s3_model_loader import load_model_from_s3
                
                for key in S3_MODEL_KEYS:
                    # Load model from S3
                    if key not in self.models or self.models[key] is None:
                        try:
                            logger.info(f"Loading diabetes model {key} from S3")
                            model_bytes = load_model_from_s3(S3_MODEL_KEYS[key], use_cache=True)
                            if isinstance(model_bytes, bytes):
                                # If S3 loader returns bytes, deserialize them
                                self.models[key] = pickle.loads(model_bytes)
                            else:
                                # If S3 loader returns the deserialized object
                                self.models[key] = model_bytes
                            logger.info(f"Diabetes model {key} loaded from S3")
                        except Exception as e:
                            logger.error(f"Failed to load diabetes model {key} from S3: {e}")
                            self.models[key] = None
                    
                    # Load columns from S3
                    if key not in self.model_columns or self.model_columns[key] is None:
                        try:
                            logger.info(f"Loading diabetes model columns {key} from S3")
                            cols_bytes = load_model_from_s3(S3_COLS_KEYS[key], use_cache=True)
                            if isinstance(cols_bytes, bytes):
                                self.model_columns[key] = pickle.loads(cols_bytes)
                            else:
                                self.model_columns[key] = cols_bytes
                            logger.info(f"Diabetes model columns {key} loaded from S3")
                        except Exception as e:
                            logger.error(f"Failed to load diabetes model columns {key} from S3: {e}")
                            self.model_columns[key] = None
                
                # If at least one model was loaded, return
                if any(self.models.values()):
                    return
            except Exception as e:
                logger.error(f"Error in S3 model loading: {e}")
        
        # Fall back to local files if needed
        for key in self.model_paths:
            # Only load if not already loaded
            if key not in self.models or self.models[key] is None:
                # Load model
                model_path = self.model_paths[key]
                try:
                    if os.path.exists(model_path):
                        with open(model_path, 'rb') as f:
                            self.models[key] = pickle.load(f)
                            logger.info(f"Diabetes model {key} loaded from local file")
                    else:
                        logger.warning(f"Diabetes model file not found: {model_path}")
                        self.models[key] = None
                except Exception as e:
                    logger.error(f"Error loading diabetes model {key}: {e}")
                    self.models[key] = None
            
            # Only load columns if not already loaded
            if key not in self.model_columns or self.model_columns[key] is None:
                # Load columns
                cols_path = self.cols_paths[key]
                try:
                    if os.path.exists(cols_path):
                        with open(cols_path, 'rb') as f:
                            self.model_columns[key] = pickle.load(f)
                            logger.info(f"Diabetes model columns {key} loaded from local file")
                    else:
                        logger.warning(f"Diabetes model columns file not found: {cols_path}")
                        self.model_columns[key] = None
                except Exception as e:
                    logger.error(f"Error loading diabetes model columns {key}: {e}")
                    self.model_columns[key] = None

    def prepare_input_df(self, participant_data, model_key):
        feature_order = [
            'age', 'weight', 'height', 'blood_type', 'gender',
            'left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_pinky',
            'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky'
        ]
        row = [participant_data.get(f) for f in feature_order]
        df = pd.DataFrame([row], columns=feature_order)
        for col in ['blood_type','gender','left_thumb','left_index','left_middle','left_ring','left_pinky',
                    'right_thumb','right_index','right_middle','right_ring','right_pinky']:
            df[col] = df[col].astype(str).str.lower()
        model_cols = self.model_columns[model_key]
        df = pd.get_dummies(df, columns=['blood_type','gender','left_thumb','left_index','left_middle','left_ring','left_pinky',
                    'right_thumb','right_index','right_middle','right_ring','right_pinky'], drop_first=True)
        for col in model_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[model_cols]
        return df
    
    def prepare_participant_data(self, participant):
        """Convert participant data to match your simple dataset format"""
        # Get all fingerprints for this participant
        fingerprints = Fingerprint.objects.filter(participant=participant)
        
        # Create dictionary with all finger positions
        finger_data = {
            'left_thumb': None,
            'left_index': None, 
            'left_middle': None,
            'left_ring': None,
            'left_pinky': None,
            'right_thumb': None,
            'right_index': None,
            'right_middle': None,
            'right_ring': None,
            'right_pinky': None
        }
        
        # Fill in the fingerprint patterns
        for fp in fingerprints:
            if fp.finger in finger_data:
                finger_data[fp.finger] = fp.pattern
        
        # Create participant row exactly matching your dataset format
        participant_data = {
            'age': participant.age,
            'weight': participant.weight,
            'height': participant.height,
            'blood_type': participant.blood_type if participant.blood_type != 'unknown' else 'UNKNOWN',
            'gender': participant.gender,
            'left_thumb': finger_data['left_thumb'],
            'left_index': finger_data['left_index'],
            'left_middle': finger_data['left_middle'],
            'left_ring': finger_data['left_ring'],
            'left_pinky': finger_data['left_pinky'],
            'right_thumb': finger_data['right_thumb'],
            'right_index': finger_data['right_index'],
            'right_middle': finger_data['right_middle'],
            'right_ring': finger_data['right_ring'],
            'right_pinky': finger_data['right_pinky']
        }
        
        return participant_data
    
    def predict_diabetes_risk(self, participant, model_key='A'):
        """Predict diabetes risk using the selected model (A or B)."""
        try:
            # Load models if not already loaded
            if not self.models or model_key not in self.models:
                self.load_models()
                
            model = self.models.get(model_key)
            model_cols = self.model_columns.get(model_key)
            
            if model is None or model_cols is None:
                return {
                    'risk': 'unknown',
                    'risk_level': 'UNKNOWN',
                    'confidence': 0.0,
                    'error': f'Model {model_key} not loaded',
                }
                
            participant_data = self.prepare_participant_data(participant)
            df = self.prepare_input_df(participant_data, model_key)
            
            # Get raw prediction
            pred = model.predict(df)[0]
            
            # Get probability scores if available
            if hasattr(model, 'predict_proba'):
                prob_scores = model.predict_proba(df)[0]
                # Assuming second column is probability of positive class
                positive_prob = prob_scores[1] if len(prob_scores) > 1 else prob_scores[0]
            else:
                # If no probabilities available, use binary prediction
                positive_prob = 1.0 if str(pred).lower() in ['diabetic', '1', 'at risk', 'risk', 'positive'] else 0.0
            
            # Get risk category based on probability
            risk_level = get_risk_category(positive_prob)
            risk_description = get_risk_description(risk_level)
            
            # Set binary classification
            risk = 'DIABETIC' if positive_prob >= 0.6 else 'HEALTHY'  # Using HIGH threshold (0.6) for binary classification
            
            return {
                'risk': risk,
                'risk_level': risk_level,
                'risk_description': risk_description,
                'confidence': round(positive_prob, 4),
                'model_used': model_key,
                'probability': round(positive_prob, 4)
            }
        except Exception as e:
            logger.error(f"Diabetes prediction failed: {str(e)}")
            return {
                'risk': 'unknown',
                'risk_level': 'UNKNOWN',
                'confidence': 0.0,
                'error': f'Prediction failed: {str(e)}',
                'model_used': model_key
            }
    