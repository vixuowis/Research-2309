from typing import *
from huggingface_hub import hf_hub_download

def load_ucf101_dataset(hf_dataset_identifier: str, filename: str) -> str:
    """
    Load a subset of the UCF-101 dataset.
    
    Args:
        hf_dataset_identifier (str): The identifier of the UCF-101 subset dataset on Hugging Face Hub.
        filename (str): The name of the file to be downloaded from the dataset.
    
    Returns:
        str: The file path of the downloaded dataset.
    """
    file_path = hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset")
    return file_path
