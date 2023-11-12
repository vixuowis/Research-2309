# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def classify_image(img_url: str):
    """
    Classify an image using a pretrained CLIP model.

    Args:
        img_url (str): The URL of the image to classify.

    Returns:
        dict: A dictionary where keys are labels and values are probabilities.
    """
    model = CLIPModel.from_pretrained('flax-community/clip-rsicd-v2')
    processor = CLIPProcessor.from_pretrained('flax-community/clip-rsicd-v2')
    image = Image.open(requests.get(img_url, stream=True).raw)
    labels = ['residential area', 'playground', 'stadium', 'forest', 'airport']
    inputs = processor(text=[f'a photo of a {l}' for l in labels], images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return {l: p.item() for l, p in zip(labels, probs[0])}

# test_function_code --------------------

def test_classify_image():
    """
    Test the classify_image function.
    """
    img_url = 'https://placekitten.com/200/300'
    result = classify_image(img_url)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(['residential area', 'playground', 'stadium', 'forest', 'airport'])
    assert all(0 <= v <= 1 for v in result.values())
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_image()