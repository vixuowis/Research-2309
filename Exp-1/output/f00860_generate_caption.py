from typing import *
from PIL import Image
import io
import requests
from transformers import AutoTokenizer
from transformers.models.fuyu.processing_fuyu import FuyuProcessor
from transformers.models.fuyu.image_processing_fuyu import FuyuImageProcessor

def generate_caption(text_prompt: str, image_url: str) -> str:
    """Generate a coco-style caption for the given image."""
    tokenizer = AutoTokenizer.from_pretrained('adept-hf-collab/fuyu-8b')
    image_processor = FuyuImageProcessor()

    processor = FuyuProcessor(image_processor=image_processor, tokenizer=tokenizer)

    image_pil = Image.open(io.BytesIO(requests.get(image_url).content))
    inputs_to_model = processor(text=text_prompt, images=image_pil)

    # Generate caption using the model
    caption = model.generate(inputs_to_model)

    return caption
