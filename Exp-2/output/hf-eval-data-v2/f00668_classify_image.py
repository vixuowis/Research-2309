# function_import --------------------

from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from PIL import Image

# function_code --------------------

def classify_image(image_path, labels):
    """
    Classify an image using the pre-trained Chinese CLIP model.

    Args:
        image_path (str): The path to the image to be classified.
        labels (list): The labels to classify the image against.

    Returns:
        dict: A dictionary where the keys are the labels and the values are the corresponding probabilities.
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
    image_path = 'test_image.jpg'
    labels = ['label1', 'label2', 'label3']
    result = classify_image(image_path, labels)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(labels)
    for label in labels:
        assert 0 <= result[label] <= 1

# call_test_function_code --------------------

test_classify_image()