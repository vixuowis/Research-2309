# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch", "Pillow"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

# function_code --------------------

def generate_image_captions(image_paths):
    """Generates captions for a list of image paths using a pre-trained model.

    Args:
        image_paths (list): A list of paths to images.

    Returns:
        list: A list of captions for the images.

    Raises:
        FileNotFoundError: When a specified image path does not exist.
        ValueError: When an image is not in RGB mode.
    """
    model = VisionEncoderDecoderModel.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
    feature_extractor = ViTImageProcessor.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
    tokenizer = AutoTokenizer.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    captions = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path {image_path} does not exist.")

        input_image = Image.open(image_path)

        if input_image.mode != 'RGB':
            raise ValueError(f"Image {image_path} is not in RGB mode.")

        pixel_values = feature_extractor(images=[input_image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        captions.append(caption.strip())

    return captions

# test_function_code --------------------

def test_generate_image_captions():
    print("Testing started.")
    # Assuming we have a function load_dataset to get a sample of images
    dataset = load_dataset("...")
    sample_data = [data['image_path'] for data in dataset[:3]]  # Extract paths for three images

    # Testing case 1: Generate captions for the images
    print("Testing case [1/1] started.")
    captions = generate_image_captions(sample_data)
    assert len(captions) == 3, f"Test case [1/1] failed: Expected 3 captions, got {len(captions)}"
    for i, caption in enumerate(captions):
        assert isinstance(caption, str), f"Test case [1/1] failed: Caption {i+1} is not a string."
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_image_captions()