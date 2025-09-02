"""
Risk threshold definitions for diabetes risk prediction
"""

# These thresholds define the probability ranges for each risk category
RISK_THRESHOLDS = {
    'LOW': (0.0, 0.3),     # 0-30% probability
    'MODERATE': (0.3, 0.6), # 30-60% probability
    'HIGH': (0.6, 0.8),     # 60-80% probability
    'VERY_HIGH': (0.8, 1.0) # 80-100% probability
}

# Risk descriptions for each category
RISK_DESCRIPTIONS = {
    'LOW': "The model predicts a low risk of diabetes. Continue maintaining a healthy lifestyle.",
    'MODERATE': "The model predicts a moderate risk of diabetes. Consider consulting with a healthcare provider for preventive measures.",
    'HIGH': "The model predicts a high risk of diabetes. It is recommended to consult with a healthcare provider for further evaluation.",
    'VERY_HIGH': "The model predicts a very high risk of diabetes. Immediate consultation with a healthcare provider is strongly recommended."
}

def get_risk_category(probability):
    """
    Convert a probability score to a risk category.
    
    Args:
        probability (float): A probability value between 0 and 1.
        
    Returns:
        str: The risk category ('LOW', 'MODERATE', 'HIGH', 'VERY_HIGH')
    """
    if probability < RISK_THRESHOLDS['LOW'][1]:
        return 'LOW'
    elif probability < RISK_THRESHOLDS['MODERATE'][1]:
        return 'MODERATE'
    elif probability < RISK_THRESHOLDS['HIGH'][1]:
        return 'HIGH'
    else:
        return 'VERY_HIGH'

def get_risk_description(risk_category):
    """
    Get the description for a risk category.
    
    Args:
        risk_category (str): Risk category ('LOW', 'MODERATE', 'HIGH', 'VERY_HIGH')
        
    Returns:
        str: Description of the risk category
    """
    return RISK_DESCRIPTIONS.get(risk_category, "Risk level unknown.")
