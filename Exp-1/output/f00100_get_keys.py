from typing import *
def get_keys(dataset):
	"""
	This function takes in a dataset and returns the keys of the first item in the dataset.

	Args:
	- dataset: A list of dictionaries representing the dataset.

	Returns:
	- keys: A list of strings representing the keys of the first item in the dataset.
	"""

	keys = []
	if len(dataset) > 0:
		keys = list(dataset[0].keys())
	return keys

