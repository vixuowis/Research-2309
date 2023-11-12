# function_import --------------------

import requests
import torch
from PIL import Image
from transformers import AlignProcessor, AlignModel

# function_code --------------------

def classify_image(image_url: str, candidate_labels: list) -> torch.Tensor:
    """
    Classify an image into one of the candidate labels using zero-shot classification.

    Args:
        image_url (str): The URL of the image to classify.
        candidate_labels (list): A list of candidate labels for classification.

    Returns:
        torch.Tensor: The probabilities of the image belonging to each candidate label.
    """
    processor = AlignProcessor.from_pretrained('kakaobrain/align-base')
    model = AlignModel.from_pretrained('kakaobrain/align-base')
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(text=candidate_labels, images=image, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return probs

# test_function_code --------------------

def test_classify_image():
    """
    Test the classify_image function.
    """
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    candidate_labels = ['an image of a cat', 'an image of a dog']
    probs = classify_image(image_url, candidate_labels)
    assert probs.shape[0] == len(candidate_labels), 'The number of probabilities should be equal to the number of candidate labels.'
    assert torch.all(probs >= 0) and torch.all(probs <= 1), 'All probabilities should be between 0 and 1.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_image()