from typing import *
from PIL import Image
import requests
from io import BytesIO

def transform_picture(url: str) -> Image.Image:
    """
    Transforms the picture so that there is a rock in there
    
    Args:
        url (str): The URL of the picture
    
    Returns:
        Image.Image: The transformed image with a rock
    """
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    rock_image = Image.open('rock.jpg')
    image.paste(rock_image, (100, 100))
    return image
