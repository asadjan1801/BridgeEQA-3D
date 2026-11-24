from google.cloud import storage
import pathlib
import hashlib
import os
from typing import Union

class GCSFileManager:
    """Manages file uploads to Google Cloud Storage for use with Gemini."""

    def __init__(self, bucket_name: str = None):
        """
        Initialize GCS file manager.

        Args:
            bucket_name: GCS bucket name. If None, uses GOOGLE_BUCKET_NAME env var.
        """
        self.storage_client = storage.Client()
        self.bucket_name = bucket_name or os.getenv("GOOGLE_BUCKET_NAME")
        self.bucket = self.storage_client.bucket(self.bucket_name)

    def upload_file(self, file_path: Union[str, pathlib.Path], folder: str = None) -> str:
        """
        Upload any file to GCS with content-based hashing to avoid duplicates.

        Args:
            file_path: Path to the file
            folder: Optional folder prefix in GCS (e.g., 'pdfs', 'images')

        Returns:
            gs:// URI for the uploaded file
        """
        file_path = pathlib.Path(file_path)

        # Read file content and create hash-based name
        file_content = file_path.read_bytes()
        content_hash = hashlib.sha256(file_content).hexdigest()[:32]

        # Determine extension and folder
        extension = file_path.suffix.lower()
        if folder is None:
            # Auto-detect folder based on extension
            if extension == '.pdf':
                folder = 'pdfs'
            elif extension in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
                folder = 'images'
            else:
                folder = 'files'

        cloud_file_name = f"{folder}/{content_hash}{extension}"

        # Check if file already exists
        blob = self.bucket.blob(cloud_file_name)
        if not blob.exists():
            # Upload the file
            blob.upload_from_filename(str(file_path))
            print(f"Uploaded {file_path.name} to GCS as {cloud_file_name}")

        # Return gs:// URI
        return f"gs://{self.bucket_name}/{cloud_file_name}"

    def get_download_url(self, file_path: Union[str, pathlib.Path], folder: str = None) -> str:
        """
        Alias for upload_file for backward compatibility.

        Args:
            file_path: Path to the file
            folder: Optional folder prefix in GCS

        Returns:
            gs:// URI for the uploaded file
        """
        return self.upload_file(file_path, folder)