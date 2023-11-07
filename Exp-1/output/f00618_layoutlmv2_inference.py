from typing import *
from transformers import pipeline

def layoutlmv2_inference(question: str, image: str) -> str:
    '''Performs inference using a finetuned LayoutLMv2 model.

    Args:
        question (str): The question to be answered.
        image (str): The path to the image file.

    Returns:
        str: The predicted answer.'''
    nlp = pipeline('question-answering', model='username/layoutlmv2-base-uncased')

    result = nlp(question=question, context=image)

    return result['answer']
