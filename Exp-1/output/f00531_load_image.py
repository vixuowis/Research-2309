from typing import *
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    image = plt.imread(image_path)
    image = np.array(image)
    return image
