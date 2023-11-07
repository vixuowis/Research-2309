from typing import *
from transformers import pipeline

def create_pipeline(task: str) -> Pipeline:
    """
    Create a pipeline for a specific inference task.
    
    Args:
        task (str): The name of the inference task.
    
    Returns:
        Pipeline: The pipeline object for the specified task.
    """
    transcriber = pipeline(task=task)
    return transcriber
