from typing import *
from huggingface_hub import Repository

def push_to_hub(self, repo_id: str) -> str:
    """
    Push the model to the Hugging Face Model Hub.
    
    Args:
        repo_id (str): The ID of the repository to push the model to.
    
    Returns:
        str: The URL of the pushed model.
    """
    repo = Repository(repo_id)
    model_path = self.save_model()
    commit_message = f"Pushing model {self.model_id}"
    repo.push_from_local_folder(model_path, commit_message)
    return repo.create_pull_request(title=commit_message, body="", labels=["Model Hub"])

