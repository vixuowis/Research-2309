from typing import *
from PIL import Image

def upscale_image(image):
    # Upscale the image
    upscaled_image = image.resize((1024, 1024))
    return upscaled_image
