from typing import *
import torch


def generate_text(image, prompt, processor, model, device):
    """Generate text based on the processed image and prompt.

    Args:
        image (PIL.Image.Image): The input image.
        prompt (str): The prompt text.
        processor: The model's processor.
        model: The text generation model.
        device: The device to run the model on.

    Returns:
        str: The generated text."""
    inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs, max_new_tokens=10)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text
