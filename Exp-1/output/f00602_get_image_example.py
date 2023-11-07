from typing import *
import matplotlib.pyplot as plt

def get_image_example(dataset, index):
    """Get an example image from the dataset.

    Args:
        dataset (dict): The dataset containing the images.
        index (int): The index of the image to retrieve.

    Returns:
        image: The example image.
    """
    image = dataset["train"][index]["image"]
    return image
