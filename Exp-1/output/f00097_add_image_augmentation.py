from typing import *
from torchvision.transforms import RandomResizedCrop, ColorJitter, Compose

def add_image_augmentation(image_processor):
	size = (
		image_processor.size['shortest_edge']
		if 'shortest_edge' in image_processor.size
		else (image_processor.size['height'], image_processor.size['width'])
	)

	_transforms = Compose([RandomResizedCrop(size), ColorJitter(brightness=0.5, hue=0.5)])
