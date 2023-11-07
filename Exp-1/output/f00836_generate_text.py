from typing import *
from transformers import pipeline

def generate_text(prompt):
    generator = pipeline(task='text-generation')
    return generator(prompt)
