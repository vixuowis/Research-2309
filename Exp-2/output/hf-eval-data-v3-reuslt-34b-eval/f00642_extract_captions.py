# function_import --------------------

import torch
import requests
from PIL import Image
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel

# function_code --------------------

def extract_captions(image_url):
    """
    Extract captions from an image using a pre-trained model from Hugging Face.

    Args:
        image_url (str): URL of the image to extract captions from.

    Returns:
        list: A list of generated captions.

    Raises:
        OSError: If there is an error in loading the image or the pre-trained model.
    """
    
    # Download the image and load it into PIL
    try:
        img = Image.open(requests.get(image_url, stream=True).raw)
        
    except OSError as e:
        print("There was an error loading the image.")
        raise
    
    # Load the pre-trained model and feature extractor
    try:
        tokenizer = AutoTokenizer.from_pretrained('google/vit-base-patch16-224')
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224', do_resize=False)
        
    except OSError as e:
        print("There was an error loading the pre-trained model.")
        raise 
    
    # Encode and tokenize the image using the feature extractor
    encoding = feature_extractor(images=img, return_tensors="pt")
    pixel_values = encoding['pixel_values']
        
    # Create a caption list and add it to our list of generated captions
    decoder = VisionEncoderDecoderModel.from_pretrained('google/vit-base-patch16-224')
    
    return [decoder.generate(pixel_values, max_length=50, num_beams=5, do_sample=True)[0]]

# test_function_code --------------------

def test_extract_captions():
    """
    Test the extract_captions function with a few test cases.
    """
    test_cases = [
        'http://images.cocodataset.org/val2017/000000039769.jpg',
        'https://placekitten.com/200/300',
        'https://placekitten.com/400/600'
    ]
    for url in test_cases:
        captions = extract_captions(url)
        assert isinstance(captions, list), 'The output should be a list.'
        assert all(isinstance(caption, str) for caption in captions), 'All elements in the output list should be strings.'
    return 'All Tests Passed'


# call_test_function_code --------------------

print(test_extract_captions())