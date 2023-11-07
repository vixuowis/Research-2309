from typing import *
import agent

def create_image(image_name):
    """Create an image with a house and car.

    Args:
        image_name (str): The name of the image.

    Returns:
        int: The return code of the image creation process.
    """
    agent.run(f"Create image: '{image_name}'", return_code=True)
