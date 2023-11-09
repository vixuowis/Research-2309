# function_import --------------------

import requests
import torch
from PIL import Image
from transformers import AlignProcessor, AlignModel

# function_code --------------------

def classify_image(url, candidate_labels):
    '''
    Classify an image into one of the given categories using the kakaobrain/align-base model.

    Args:
        url (str): The URL of the image to classify.
        candidate_labels (list): A list of strings representing the possible categories for the image.

    Returns:
        torch.Tensor: A tensor representing the probabilities of the image belonging to each category.
    '''
    processor = AlignProcessor.from_pretrained('kakaobrain/align-base')
    model = AlignModel.from_pretrained('kakaobrain/align-base')
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(text=candidate_labels, images=image, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return probs

# test_function_code --------------------

def test_classify_image():
    '''
    Test the classify_image function with a sample image and categories.
    '''
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    candidate_labels = ['an image of casual dressing', 'an image of formal dressing']
    probs = classify_image(url, candidate_labels)
    assert probs.shape == (1, len(candidate_labels)), 'Output shape is incorrect'
    assert torch.all(probs >= 0) and torch.all(probs <= 1), 'Probabilities are not between 0 and 1'
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5), 'Probabilities do not sum to 1'

# call_test_function_code --------------------

test_classify_image()