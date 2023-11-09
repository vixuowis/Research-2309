# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def classify_image_content(image_url):
    """
    Classify an image's content and check if it contains a cat or a dog using the pretrained CLIP model 'openai/clip-vit-base-patch16'.

    Args:
        image_url (str): The URL of the image to be classified.

    Returns:
        dict: A dictionary with the probabilities of the image containing a cat or a dog.
    """
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(text=['a photo of a cat', 'a photo of a dog'], images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return {'cat': probs[0].item(), 'dog': probs[1].item()}

# test_function_code --------------------

def test_classify_image_content():
    """
    Test the function classify_image_content.
    """
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    result = classify_image_content(image_url)
    assert isinstance(result, dict)
    assert 'cat' in result
    assert 'dog' in result
    assert 0 <= result['cat'] <= 1
    assert 0 <= result['dog'] <= 1

# call_test_function_code --------------------

test_classify_image_content()