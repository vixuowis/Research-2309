from typing import *
from transformers import pipeline

def pipeline(model: str, device_map: str):
    transcriber = pipeline(model=model, device_map=device_map)
