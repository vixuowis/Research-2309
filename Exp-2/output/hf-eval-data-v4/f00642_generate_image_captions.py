# requirements_file --------------------

!pip install -U torch, requests, PIL.Image, transformers

# function_import --------------------

import torch
import requests
from PIL import Image
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel

# function_code --------------------

def generate_image_captions(image_path):
    loc = 'ydshieh/vit-gpt2-coco-en'
    feature_extractor = ViTFeatureExtractor.from_pretrained(loc)
    tokenizer = AutoTokenizer.from_pretrained(loc)
    model = VisionEncoderDecoderModel.from_pretrained(loc)
    model.eval()

    with Image.open(requests.get(image_path, stream=True).raw) as image:
        pixel_values = feature_extractor(images=image, return_tensors='pt').pixel_values

    with torch.no_grad():
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True).sequences
    captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    captions = [caption.strip() for caption in captions]

    return captions

# test_function_code --------------------

def test_generate_image_captions():
    print("Testing generate_image_captions function.")
    sample_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    captions = generate_image_captions(sample_image_url)
    assert isinstance(captions, list), "The output should be a list of captions."
    assert all(isinstance(caption, str) for caption in captions), "Each caption should be a string."
    print("Test passed.")

test_generate_image_captions()