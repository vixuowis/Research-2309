# requirements_file --------------------

!pip install -U transformers torch PIL

# function_import --------------------

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import os

# function_code --------------------

def generate_image_captions(image_paths, model_name='nlpconnect/vit-gpt2-image-captioning'):
    """
    Generate captions for a list of image paths using a pre-trained image captioning model.

    Parameters:
        image_paths (list of str): A list of file paths to the images.
        model_name (str): The name of the pre-trained model to use.

    Returns:
        list of str: A list of generated captions corresponding to the input images.
    """
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    captions = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            captions.append(f"File not found: {image_path}")
            continue

        input_image = Image.open(image_path)
        if input_image.mode != "RGB":
            input_image = input_image.convert(mode="RGB")

        pixel_values = feature_extractor(images=[input_image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        output_ids = model.generate(pixel_values, **gen_kwargs)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        captions.append(caption)

    return captions

# test_function_code --------------------

def test_generate_image_captions():
    print("Testing function: generate_image_captions")

    # Test case: generating captions for existing images
    print("Testing case [1/3] - Generating captions for existing images.")
    existing_images = ['sample_image_1.jpg', 'sample_image_2.jpg']
    captions = generate_image_captions(existing_images)
    assert len(captions) == 2, f"Test case [1/3] failed: Expected 2 captions, got {len(captions)}"
    print("Test case [1/3] succeeded.")

    # Test case: handling non-existent file paths
    print("Testing case [2/3] - Handling non-existent file paths.")
    non_existent_images = ['non_existing_image_1.jpg']
    captions = generate_image_captions(non_existent_images)
    assert captions[0] == f"File not found: {non_existent_images[0]}", f"Test case [2/3] failed: Did not handle non-existent file path correctly"
    print("Test case [2/3] succeeded.")

    # Test case: generating captions for multiple images including non-existent ones
    print("Testing case [3/3] - Generating captions for multiple images including non-existent ones.")
    mixed_images = ['sample_image_1.jpg', 'non_existing_image_2.jpg']
    captions = generate_image_captions(mixed_images)
    assert len(captions) == 2, f"Test case [3/3] failed: Expected 2 results, got {len(captions)}"
    assert captions[1] == f"File not found: non_existing_image_2.jpg", f"Test case [3/3] failed: Did not handle non-existent file path correctly"
    print("Test case [3/3] succeeded.")
    print("Testing finished.")

test_generate_image_captions()