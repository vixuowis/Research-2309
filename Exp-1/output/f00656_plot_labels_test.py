from f00656_plot_labels import *
def test_plot_labels():
	import numpy as np
	
	# Test case 1
	labels = np.random.rand(80, 100)
	plot_labels(labels)

	# Test case 2
	labels = np.zeros((80, 100))
	plot_labels(labels)

	# Test case 3
	labels = np.ones((80, 100))
	plot_labels(labels)

