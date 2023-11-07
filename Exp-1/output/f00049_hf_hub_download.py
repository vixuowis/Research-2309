from typing import *
from huggingface_hub import hf_hub_download

def hf_hub_download(repo_id: str, filename: str, cache_dir: str) -> None:
    """Download a file from the Hugging Face Hub to a specific path.

    Args:
        repo_id (str): The repository ID in the format `owner/repo_name`.
        filename (str): The name of the file to download.
        cache_dir (str): The path to the directory where the file should be saved.

    Returns:
        None
    """
    hf_hub_download(repo_id, filename, cache_dir)
