from typing import *
from transformers import AutoImageProcessor

def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    # Check if the pretrained model name or path is valid
    if not is_valid_model_name_or_path(pretrained_model_name_or_path):
        raise ValueError(
            f"Invalid pretrained model name or path: {pretrained_model_name_or_path}."
        )

    # Load the model configuration
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

    # Instantiate the image processor
    image_processor = cls(config, *model_args, **kwargs)

    # Load the weights of the image processor
    image_processor.load_pretrained(pretrained_model_name_or_path)

    return image_processor
