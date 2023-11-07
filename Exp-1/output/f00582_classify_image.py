from typing import *
from PIL import Image
import requests

def classify_image(url):
	image = Image.open(requests.get(url, stream=True).raw)
	return image
