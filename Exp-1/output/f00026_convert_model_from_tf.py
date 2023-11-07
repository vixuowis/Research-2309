from typing import *
def convert_model_from_tf(tf_save_directory):
	"""
	Converts a TensorFlow model to a PyTorch model.

	Args:
		tf_save_directory (str): The directory where the TensorFlow model is saved.

	Returns:
		pt_model (AutoModelForSequenceClassification): The converted PyTorch model.
	"""

	tokenizer = AutoTokenizer.from_pretrained(tf_save_directory)
	pt_model = AutoModelForSequenceClassification.from_pretrained(tf_save_directory, from_tf=True)
	return pt_model

