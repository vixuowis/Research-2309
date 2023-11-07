from typing import *
from transformers import AutoModel

def sharded_checkpoints(max_shard_size: int, model_name: str) -> None:
    """
    This function demonstrates how to use sharded checkpoints in the transformers library.

    Args:
        max_shard_size (int): The maximum size (in GB) before sharding the checkpoints.
        model_name (str): The name of the model to load.

    Returns:
        None
    """
    model = AutoModel.from_pretrained(model_name)
