"""Helpers for uploading, deleting, and locating OpenAI File API resources."""

from pathlib import Path
from typing import Iterable, Optional
import os
import time
import hashlib
from functools import lru_cache, wraps
from pydantic import BaseModel
from openai import OpenAI
import logfire

from openai.types import FileObject, FilePurpose

logfire.warn("The file_utils module will be deprecated")

def timed_lru_cache(seconds: int, maxsize: Optional[int] = None):
    """Decorator that adds time-based expiration to lru_cache.

    Args:
        seconds: Cache lifetime in seconds
        maxsize: Maximum number of cached results (None = unlimited)
    """

    def wrapper_cache(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = seconds
        func.expiration = time.time() + seconds

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if time.time() >= func.expiration:
                func.cache_clear()
                func.expiration = time.time() + seconds
            return func(*args, **kwargs)

        wrapped_func.cache_info = func.cache_info
        wrapped_func.cache_clear = func.cache_clear
        return wrapped_func

    return wrapper_cache


def get_file_hash_prefix(file_path: str) -> str:
    """Generate a hash prefix from the entire file content.

    Args:
        file_path: Path to the file to hash

    Returns:
        First 64 characters of the SHA-256 hash
    """
    hasher = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files efficiently
        while chunk := f.read(8192):
            hasher.update(chunk)

    # Get hex digest and return first 64 characters
    return hasher.hexdigest()


def upload_file_to_openai(
    file_path: str,
    *,
    purpose: FilePurpose = "user_data",
) -> FileObject:
    """Upload a local file to the OpenAI Files API.

    Files are uploaded with directory prefixes to avoid naming conflicts
    across different directories.

    Args:
        file_path: Path to the local file to upload.
        purpose: Optional OpenAI "purpose" tag for the file. Defaults to ``"user_data"``
            which is accepted by the Assistants/Responses APIs.

    Returns:
        The ID of the uploaded file returned by OpenAI.
    """
    client = OpenAI()
    path = Path(file_path)

    # Generate hash prefix from file content
    hash_prefix = get_file_hash_prefix(file_path)
    # Create filename with hash prefix and preserve the original extension
    prefixed_filename = f"{hash_prefix}{path.suffix}"

    with open(file_path, "rb") as file_stream:
        # Create a file-like object with the prefixed filename
        file = client.files.create(
            file=(prefixed_filename, file_stream),
            purpose=purpose,
        )
    return file


def delete_file_from_openai(file_id: str) -> bool:
    """Delete a file from the OpenAI Files API by its ID.

    Args:
        file_id: The ID of the file to delete.

    Returns:
        ``True`` if OpenAI confirmed the deletion, otherwise ``False``.
    """
    client = OpenAI()
    response = client.files.delete(file_id)
    return bool(getattr(response, "deleted", False))


@timed_lru_cache(seconds=10, maxsize=None)
def list_openai_files(*, purpose: Optional[FilePurpose] = None) -> Iterable[FileObject]:
    """Return the files currently stored in the OpenAI account.

    This function is cached for 10 seconds to reduce API calls. The cache is
    automatically cleared after expiration.

    Args:
        purpose: Optional OpenAI "purpose" filter (for example, ``"assistants"`` or
            ``"fine-tune"``). If omitted, returns all files.

    Returns:
        An iterable of ``OpenAIFile`` objects as returned by the SDK.
    """
    client = OpenAI()
    if purpose is not None:
        files_page = client.files.list(purpose=purpose)
    else:
        files_page = client.files.list()
    # Convert to list to ensure it's cached properly
    return list(files_page)


def find_openai_file(
    filename: str,
    *,
    purpose: Optional[FilePurpose] = None,
) -> Optional[FileObject]:
    """Locate an uploaded file by name.

    Files are expected to be uploaded with hash prefixes in the format:
    {hash_prefix}_{filename}

    Args:
        filename: The file path or name to search for.
        purpose: Optional purpose filter mirroring ``list_openai_files``.

    Returns:
        The matching ``OpenAIFile`` instance if found, otherwise ``None``.
    """
    path = Path(filename)

    # Check if the file exists locally to generate hash
    if path.exists():
        # Generate hash prefix from file content
        hash_prefix = get_file_hash_prefix(str(path))
        prefixed_filename = f"{hash_prefix}{path.suffix}"
    else:
        raise FileNotFoundError(f"File {path} does not exist")

    for file in list_openai_files(purpose=purpose):
        file_name = getattr(file, "filename", None)
        if file_name == prefixed_filename:
            return file

    return None


def upload_file_to_openai_if_not_exists(
    file_path: str,
    *,
    purpose: FilePurpose = "user_data",
) -> FileObject:
    file = find_openai_file(file_path, purpose=purpose)
    if not file:
        return upload_file_to_openai(file_path, purpose=purpose)
    return file


def find_openai_files_from_paths(
    file_paths: Iterable[str], purpose: FilePurpose = "user_data"
) -> list[FileObject]:
    """Find multiple OpenAI files from a collection of file paths.

    This function now calls the OpenAI API only once and builds a lookup
    dictionary for efficient searching. Files are expected to be uploaded
    with hash prefixes in the format: {hash_prefix}_{filename}

    Args:
        file_paths: An iterable of file paths to find in OpenAI
        purpose: The OpenAI purpose tag for the files (default: "user_data")

    Returns:
        A list of FileObject instances for each found file
    """
    # Get all files once - single API call
    all_files = list_openai_files(purpose=purpose)

    file_paths_list = list(file_paths)
    if not file_paths_list:
        return []

    # Build lookup dictionary for all OpenAI files
    file_lookup = {}
    for file in all_files:
        filename = getattr(file, "filename", None)
        if filename:
            file_lookup[filename] = file

    # Find each requested file in the lookup dictionary
    found_files = []
    for file_path in file_paths_list:
        path = Path(file_path)

        # Check if the file exists locally to generate hash
        if path.exists():
            # Generate hash prefix from file content
            hash_prefix = get_file_hash_prefix(str(path))
            prefixed_filename = f"{hash_prefix}{path.suffix}"

            # Look for the file with hash prefix
            file_obj = file_lookup.get(prefixed_filename)
            if file_obj:
                found_files.append(file_obj)

    return found_files
