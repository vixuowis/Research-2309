from typing import *
from PIL import Image
import requests
from io import BytesIO

def draw_and_transform_picture():
    # Draw a picture of the sea
    sea_image = Image.new('RGB', (500, 500), 'blue')

    # Transform the picture by adding an island
    response = requests.get('https://example.com/island.jpg')
    island_image = Image.open(BytesIO(response.content))
    sea_with_island = Image.alpha_composite(sea_image.convert('RGBA'), island_image)

    return sea_with_island
