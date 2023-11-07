from typing import *
import numpy as np
import matplotlib.pyplot as plt

def show_image(img):
    '''
    Show the image using matplotlib.

    Parameters:
        img (Tensor): The image tensor to be shown.

    Returns:
        None
    '''
    plt.imshow(img.permute(1, 2, 0))
