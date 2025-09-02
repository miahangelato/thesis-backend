import logging
import os

# Configure module-level logger
logger = logging.getLogger('core')

# Is the app running on Railway?
IS_PRODUCTION = os.environ.get('RAILWAY_DEPLOYMENT') == 'True'

def debug(message):
    """Log debug message - skips in production to reduce log volume"""
    if not IS_PRODUCTION:
        logger.debug(message)

def info(message):
    """Log info message - skips in production to reduce log volume"""
    if not IS_PRODUCTION:
        logger.info(message)

def warning(message):
    """Log warning message - always logged"""
    logger.warning(message)

def error(message):
    """Log error message - always logged"""
    logger.error(message)

def critical(message):
    """Log critical message - always logged"""
    logger.critical(message)

# Alias for backward compatibility with print statements
def print_debug(message):
    """Replacement for print("[DEBUG] ...") statements"""
    debug(message)
