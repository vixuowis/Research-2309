from typing import *
import matplotlib.pyplot as plt

def plot_labels(labels):
	plt.figure()
	plt.imshow(labels.T)
	plt.show()
