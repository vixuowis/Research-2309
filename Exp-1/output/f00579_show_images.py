from typing import *
import matplotlib.pyplot as plt

def show_images(image_target, query_image):
	"""
	Display two images side by side.

	Params:
	- image_target: The target image to display.
	- query_image: The query image to display.
	"""
	fig, ax = plt.subplots(1, 2)
	ax[0].imshow(image_target)
	ax[1].imshow(query_image)
