from typing import *
from transformers import pipeline

def video_classification_pipeline(model: str) -> Any:
    """Instantiate a pipeline for video classification with the given model.

    Args:
        model (str): The name or path of the pre-trained video classification model.

    Returns:
        Any: The video classification pipeline.
    """
    video_cls = pipeline(model=model)
    return video_cls
