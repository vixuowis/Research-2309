from typing import *
from PIL import Image, ImageDraw

def make_image():
    """Create an image of a house and a car

    Args:
        None

    Returns:
        PIL.Image.Image: The generated image
    """
    image = Image.new('RGB', (500, 500), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    # Draw house
    # Draw car
    return image
