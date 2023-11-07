from typing import *
from transformers import pipeline

def run_question_answering(question, context, model):
    question_answerer = pipeline("question-answering", model=model)
    result = question_answerer(question=question, context=context)
    return result
