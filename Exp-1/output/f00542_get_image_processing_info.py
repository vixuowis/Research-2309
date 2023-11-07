from typing import *
from torchvision.transforms import Normalize

def get_image_processing_info(image_processor, model):
	"""
	Get image processing information from the pre-trained model.

	Args:
	    image_processor (object): The image processor associated with the pre-trained model.
	    model (object): The pre-trained model.

	Returns:
	    dict: A dictionary containing the image processing information.
	"""
	mean = image_processor.image_mean
	std = image_processor.image_std

	if "shortest_edge" in image_processor.size:
		height = width = image_processor.size["shortest_edge"]
	else:
		height = image_processor.size["height"]
		width = image_processor.size["width"]

	resize_to = (height, width)

	num_frames_to_sample = model.config.num_frames
	sample_rate = 4
	fps = 30
	clip_duration = num_frames_to_sample * sample_rate / fps

	return {
		"mean": mean,
		"std": std,
		"resize_to": resize_to,
		"num_frames_to_sample": num_frames_to_sample,
		"sample_rate": sample_rate,
		"fps": fps,
		"clip_duration": clip_duration
	}
