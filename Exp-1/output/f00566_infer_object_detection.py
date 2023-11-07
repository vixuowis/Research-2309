from typing import *
from transformers import pipeline
import requests
from PIL import Image

def infer_object_detection(url):
    image = Image.open(requests.get(url, stream=True).raw)
    obj_detector = pipeline("object-detection", model="devonho/detr-resnet-50_finetuned_cppe5")
    return obj_detector(image)
