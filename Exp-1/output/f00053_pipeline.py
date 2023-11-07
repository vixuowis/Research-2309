from typing import *
from transformers import pipeline


def pipeline(model: str) -> Callable[[str], Dict[str, str]]:
    '''Transcribes speech to text using the specified model.

    Args:
        model (str): The name or path of the pre-trained ASR model.

    Returns:
        Callable[[str], Dict[str, str]]: The transcriber function.
    '''
    
    def transcriber(url: str) -> Dict[str, str]:
        transcriber = pipeline(model=model)
        result = transcriber(url)
        return result
    
    return transcriber
