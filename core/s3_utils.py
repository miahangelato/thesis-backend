"""
S3 File Access Utility
Handles secure access to S3 files for processing
"""

import boto3
import tempfile
import requests
import os
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

class S3FileHandler:
    """Utility class for handling S3 file access"""
    
    def __init__(self):
        self.s3_client = None
        self._init_s3_client()
    
    def _init_s3_client(self):
        """Initialize S3 client"""
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_S3_REGION_NAME
            )
            logger.info("✅ S3 client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize S3 client: {e}")
            self.s3_client = None
    
    def get_signed_url(self, s3_key, expiration=3600):
        """Generate a signed URL for S3 object"""
        if not self.s3_client:
            raise Exception("S3 client not initialized")
        
        try:
            # Handle the media prefix issue
            # If the key doesn't start with 'media/', add it
            if not s3_key.startswith('media/'):
                s3_key = f"media/{s3_key}"
            
            signed_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': settings.AWS_STORAGE_BUCKET_NAME,
                    'Key': s3_key
                },
                ExpiresIn=expiration
            )
            logger.info(f"✅ Generated signed URL for: {s3_key}")
            return signed_url
        except Exception as e:
            logger.error(f"❌ Failed to generate signed URL for {s3_key}: {e}")
            raise
    
    def download_to_temp_file(self, s3_key, suffix='.png'):
        """Download S3 file to temporary file and return path"""
        try:
            # Generate signed URL
            signed_url = self.get_signed_url(s3_key)
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            
            # Download file
            response = requests.get(signed_url, timeout=30)
            response.raise_for_status()
            
            # Write to temp file
            temp_file.write(response.content)
            temp_file.flush()
            temp_file.close()
            
            logger.info(f"✅ Downloaded {s3_key} to temp file: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"❌ Failed to download {s3_key}: {e}")
            raise
    
    def cleanup_temp_file(self, temp_file_path):
        """Clean up temporary file"""
        try:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                logger.info(f"✅ Cleaned up temp file: {temp_file_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup temp file {temp_file_path}: {e}")

# Global instance
s3_handler = S3FileHandler()
