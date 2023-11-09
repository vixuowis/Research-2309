# function_import --------------------

import torch
import requests
from PIL import Image
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel

# function_code --------------------

loc = 'ydshieh/vit-gpt2-coco-en'
feature_extractor = ViTFeatureExtractor.from_pretrained(loc)
tokenizer = AutoTokenizer.from_pretrained(loc)
model = VisionEncoderDecoderModel.from_pretrained(loc)
model.eval()

def extract_captions(image):
    """
    This function extracts captions from an image using a pre-trained model from Hugging Face.

    Args:
        image (PIL.Image): The image from which to extract captions.

    Returns:
        list: A list of captions extracted from the image.
    """
    pixel_values = feature_extractor(images=image, return_tensors='pt').pixel_values
    with torch.no_grad():
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True).sequences
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

# test_function_code --------------------

def test_extract_captions():
    """
    This function tests the extract_captions function by comparing the output with expected results.
    """
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    with Image.open(requests.get(url, stream=True).raw) as image:
        preds = extract_captions(image)
    assert isinstance(preds, list), 'The output should be a list.'
    assert all(isinstance(pred, str) for pred in preds), 'All elements in the output list should be strings.'

# call_test_function_code --------------------

test_extract_captions()