from typing import *
import skimage
import numpy as np
from PIL import Image

def load_image():
    """
    Loads and converts the astronaut image.

    Returns:
        PIL.Image: The loaded and converted image.
    """
    image = skimage.data.astronaut()
    image = Image.fromarray(np.uint8(image)).convert("RGB")

    return image
