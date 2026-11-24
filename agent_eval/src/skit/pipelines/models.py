"""
Models for pipeline data structures.
"""

from pydantic import BaseModel


class ExtractedImage(BaseModel):
    image_path: str
    cloud_download_url: str