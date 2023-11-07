from typing import *
import matplotlib.pyplot as plt

def visualize_spectrogram(spectrogram):
	plt.figure()
	plt.imshow(spectrogram.T)
	plt.show()
