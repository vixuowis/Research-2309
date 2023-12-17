# requirements_file --------------------

import subprocess

requirements = ["requests", "torch", "Pillow", "transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import requests
import torch
from PIL import Image
from transformers import AlignProcessor, AlignModel

# function_code --------------------

def classify_image_with_dress_options(image_url, candidate_labels):
    """
    Classify an image into specified dress categories using zero-shot classification.

    Args:
        image_url (str): URL of the image to classify.
        candidate_labels (list of str): A list of dress options (categories) to classify the image into.

    Returns:
        dict: Probabilities for each category in candidate_labels.

    Raises:
        ValueError: If the image cannot be fetched from the URL.
    """
    processor = AlignProcessor.from_pretrained('kakaobrain/align-base')
    model = AlignModel.from_pretrained('kakaobrain/align-base')

    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except requests.RequestException:
        raise ValueError('Image could not be fetched from the URL.')

    inputs = processor(text=candidate_labels, images=image, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    return {label: prob for label, prob in zip(candidate_labels, probs[0].tolist())}

# test_function_code --------------------

def test_classify_image_with_dress_options():
    print("Testing started.")
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    candidate_labels = ['an image of casual dressing', 'an image of formal dressing', 'an image of sport dressing']

    # Testing case [1/3]
    print("Testing case [1/3] started.")
    try:
        result = classify_image_with_dress_options(image_url, candidate_labels)
        assert isinstance(result, dict), "Output should be a dictionary."
        assert set(result.keys()) == set(candidate_labels), "Keys of the result should match candidate_labels."
    except ValueError as e:
        assert False, f"Test case [1/3] failed: {e}"

    # Testing case [2/3]
    print("Testing case [2/3] started.")
    invalid_image_url = 'http://invalid-url.com'
    try:
        classify_image_with_dress_options(invalid_image_url, candidate_labels)
        assert False, "Test case [2/3] should have raised ValueError."
    except ValueError:
        pass  # This is expected

    # Testing case [3/3]
    print("Testing case [3/3] started.")
    empty_labels = []
    try:
        result_empty = classify_image_with_dress_options(image_url, empty_labels)
        assert result_empty == {}, "Output should be an empty dictionary when no candidate_labels are provided."
    except Exception as e:
        assert False, f"Test case [3/3] failed: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_classify_image_with_dress_options()