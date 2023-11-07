from typing import *
from transformers import ViTImageProcessor

def create_custom_image_processor(resample, do_normalize, image_mean):
    return ViTImageProcessor(resample=resample, do_normalize=do_normalize, image_mean=image_mean)

my_vit_extractor = create_custom_image_processor(resample="PIL.Image.BOX", do_normalize=False, image_mean=[0.3, 0.3, 0.3])
print(my_vit_extractor)
