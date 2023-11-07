from typing import *
from typing import Iterator
from transformers import pipeline

def run_pipeline_on_dataset(data: Iterator[str], model: str, device: int) -> int:
    '''
    Run a pipeline on a dataset and return the total number of generated characters.

    Args:
        data (Iterator[str]): An iterator that yields strings.
        model (str): The name of the pre-trained model to use.
        device (int): The device index to run the pipeline on.

    Returns:
        int: The total number of generated characters.
    '''
    pipe = pipeline(model=model, device=device)
    generated_characters = 0
    for out in pipe(data):
        generated_characters += len(out[0]['generated_text'])
    return generated_characters
