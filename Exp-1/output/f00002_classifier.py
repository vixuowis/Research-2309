from typing import *
from transformers import pipeline
from typing import List, Dict, Union


def classifier(text: str) -> List[Dict[str, Union[str, float]]]:
    """Perform sentiment analysis on the input text.

    Args:
        text (str): The input text to analyze.

    Returns:
        List[Dict[str, Union[str, float]]]: A list of dictionaries containing the label and score of the sentiment analysis.
    """
    classifier = pipeline('sentiment-analysis')
    result = classifier(text)
    return result
