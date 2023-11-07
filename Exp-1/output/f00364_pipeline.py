from typing import *
from transformers import pipeline

translator = pipeline("translation", model="my_awesome_opus_books_model")
translator(text)
