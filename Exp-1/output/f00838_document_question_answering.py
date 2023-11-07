from typing import *
from transformers import pipeline
from PIL import Image
import requests

def document_question_answering(url: str, question: str) -> dict:
    """
    Performs document question answering on an image of a document.

    Args:
        url (str): The URL of the image of the document.
        question (str): The question about the document.

    Returns:
        dict: A dictionary containing the answer, score, start position, and end position of the answer.
    """
    image = Image.open(requests.get(url, stream=True).raw)
    doc_question_answerer = pipeline("document-question-answering", model="magorshunov/layoutlm-invoices")
    preds = doc_question_answerer(question=question, image=image)
    return preds[0]
