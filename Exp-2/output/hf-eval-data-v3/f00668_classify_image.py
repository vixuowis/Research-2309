# function_import --------------------

from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from PIL import Image

# function_code --------------------

def classify_image(image_path, labels):
    """
    Classify an image based on the given labels using a pre-trained Chinese CLIP model.

    Args:
        image_path (str): The path to the image to be classified.
        labels (list): A list of labels for classification.

    Returns:
        dict: A dictionary with labels as keys and corresponding probabilities as values.
    """
    model = ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14-336px')
    processor = ChineseCLIPProcessor.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14-336px')
    image = Image.open(image_path)
    inputs = processor(images=image, text=labels, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return {label: prob for label, prob in zip(labels, probs.tolist())}

# test_function_code --------------------

def test_classify_image():
    """
    Test the classify_image function.
    """
    image_path = 'https://placekitten.com/200/300'
    labels = ['cat', 'dog', 'bird']
    result = classify_image(image_path, labels)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert set(result.keys()) == set(labels), 'The keys of the result should match the labels.'
    assert all(isinstance(value, float) for value in result.values()), 'The values of the result should be probabilities.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_image()