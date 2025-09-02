"""
Backend encryption utilities that match frontend encryption
"""
import os
import base64
from cryptography.fernet import Fernet
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class BackendDecryption:
    def __init__(self):
        self.fernet = None
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize the Fernet encryption with the loaded key."""
        try:
            key = self._load_or_generate_key()
            self.fernet = Fernet(key)
            logger.info("✅ Encryption initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            raise
    
    def _load_or_generate_key(self):
        """Load encryption key from environment variable."""
        # Production approach: Only use environment variable
        key_base64 = os.environ.get('ENCRYPTION_KEY')
        if key_base64:
            try:
                return base64.b64decode(key_base64)
            except Exception as e:
                logger.error(f"Invalid base64 key in environment: {e}")
                raise ValueError("Invalid encryption key format")
        
        # For development only: Generate a temporary key and warn
        if settings.DEBUG:
            logger.warning("⚠️  No ENCRYPTION_KEY found! Using temporary key for development.")
            logger.warning("⚠️  Add ENCRYPTION_KEY to your .env file for production!")
            return Fernet.generate_key()
        else:
            raise ValueError("ENCRYPTION_KEY environment variable is required in production")
    
    def generate_new_key(self):
        """Generate a new encryption key and save to environment"""
        key = Fernet.generate_key()
        self.fernet = Fernet(key)
        
        # Encode to base64 for environment storage
        encoded_key = base64.b64encode(key).decode('utf-8')
        os.environ['ENCRYPTION_KEY'] = encoded_key
        
        logger.info("✅ New encryption key generated")
        logger.warning("⚠️ Add this to your .env file:")
        logger.warning(f"ENCRYPTION_KEY={encoded_key}")
    
    def generate_temp_key(self):
        """Generate a temporary key for development only"""
        key = Fernet.generate_key()
        self.fernet = Fernet(key)
        logger.warning("🔧 Using temporary encryption key - data won't persist!")
    
    def decrypt_fingerprint_data(self, encrypted_data):
        """Decrypt fingerprint data received from frontend"""
        if not self.fernet:
            raise Exception("Decryption not initialized")
        
        try:
            # Handle both string and bytes input
            if isinstance(encrypted_data, str):
                encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            else:
                encrypted_bytes = encrypted_data
            
            # Decrypt
            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
            
            return decrypted_bytes
        
        except Exception as e:
            logger.error(f"❌ Fingerprint decryption failed: {e}")
            raise Exception("Failed to decrypt fingerprint data")
    
    def decrypt_participant_data(self, encrypted_data):
        """Decrypt participant data"""
        if not self.fernet:
            raise Exception("Decryption not initialized")
        
        try:
            # Handle both string and bytes input
            if isinstance(encrypted_data, str):
                encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            else:
                encrypted_bytes = encrypted_data
            
            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
            
            return decrypted_bytes.decode('utf-8')
        
        except Exception as e:
            logger.error(f"❌ Participant data decryption failed: {e}")
            raise Exception("Failed to decrypt participant data")
    
    def encrypt_data(self, data):
        """Encrypt data (convenience method)"""
        if not self.fernet:
            raise Exception("Encryption not initialized")
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted_data = self.fernet.encrypt(data)
        return base64.b64encode(encrypted_data).decode('utf-8')

    def decrypt_form_data(self, form_data):
        """Decrypt form data received from frontend"""
        if not self.fernet:
            logger.warning("⚠️ Decryption not initialized - returning data as-is")
            return form_data
        
        decrypted_data = {}
        
        try:
            for key, value in form_data.items():
                # Skip non-string values and short strings that aren't encrypted
                if not isinstance(value, str) or len(value) < 50:
                    decrypted_data[key] = value
                    continue
                
                try:
                    # Try to decrypt the value
                    encrypted_bytes = base64.b64decode(value.encode('utf-8'))
                    decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
                    decrypted_value = decrypted_bytes.decode('utf-8')
                    decrypted_data[key] = decrypted_value
                    logger.info(f"✅ Decrypted field: {key}")
                
                except Exception as decrypt_error:
                    # If decryption fails, assume it's not encrypted
                    logger.warning(f"⚠️ Could not decrypt {key}: {decrypt_error} - using original value")
                    decrypted_data[key] = value
            
            logger.info(f"✅ Form data decryption completed - {len(decrypted_data)} fields processed")
            return decrypted_data
        
        except Exception as e:
            logger.error(f"❌ Form data decryption failed: {e}")
            # Return original data if decryption fails completely
            logger.warning("⚠️ Returning original form data due to decryption failure")
            return form_data

# Global instance
backend_decryption = BackendDecryption()
