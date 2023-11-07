from typing import *
from transformers import PreTrainedModel

def push_to_hub(self, repo_name: str) -> str:
    """
    Upload the model to the Hugging Face Model Hub.

    Args:
        repo_name (str): The name of the repository to which the model should be uploaded.

    Returns:
        str: The URL of the uploaded model on the Hugging Face Model Hub.
    """
    return self.push_to_hub_model(repo_name)
