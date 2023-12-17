# requirements_file --------------------

!pip install -U torch requests Pillow transformers

# function_import --------------------

import torch
import requests
from PIL import Image
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel

# function_code --------------------

def generate_image_caption(image_url: str) -> str:
    """Generate a caption for the image at the specified URL.

    Args:
        image_url (str): The URL of the image to caption.

    Returns:
        str: The generated caption for the image.

    Raises:
        ValueError: If the image_url is not valid.
    """
    feature_extractor = ViTFeatureExtractor.from_pretrained('ydshieh/vit-gpt2-coco-en')
    tokenizer = AutoTokenizer.from_pretrained('ydshieh/vit-gpt2-coco-en')
    model = VisionEncoderDecoderModel.from_pretrained('ydshieh/vit-gpt2-coco-en')
    model.eval()

    try:
        with Image.open(requests.get(image_url, stream=True).raw) as image:
            pixel_values = feature_extractor(images=image, return_tensors='pt').pixel_values
            with torch.no_grad():
                output_ids = model.generate(pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True).sequences
            captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            caption = captions[0].strip()
            return caption
    except Exception as e:
        raise ValueError('Invalid image URL or error processing the image.') from e

# test_function_code --------------------

def test_generate_image_caption():
    print("Testing started.")
    test_urls = [
        'http://images.cocodataset.org/val2017/000000039769.jpg',
        'https://invalid-url.com/image.jpg',
        'http://images.cocodataset.org/val2017/000000285149.jpg'
    ]

    # Test case 1: Valid image URL with expected caption.
    print("Testing case [1/3] started.")
    caption = generate_image_caption(test_urls[0])
    assert caption is not None and len(caption) > 0, "Test case [1/3] failed: Expected a non-empty caption."

    # Test case 2: Invalid image URL.
    print("Testing case [2/3] started.")
    try:
        _ = generate_image_caption(test_urls[1])
        assert False, "Test case [2/3] failed: Expected a ValueError for an invalid URL."
    except ValueError:
        pass

    # Test case 3: Valid image URL with expected caption.
    print("Testing case [3/3] started.")
    caption = generate_image_caption(test_urls[2])
    assert caption is not None and len(caption) > 0, "Test case [3/3] failed: Expected a non-empty caption."
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_image_caption()