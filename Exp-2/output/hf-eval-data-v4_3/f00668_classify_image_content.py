# requirements_file --------------------

import subprocess

requirements = ["Pillow", "transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# function_code --------------------

def classify_image_content(image_path, labels):
    """
    Classify the content of an image using a pre-trained Chinese CLIP model.

    Args:
        image_path (str): The file path of the image to classify.
        labels (list of str): A list of labels for zero-shot classification.

    Returns:
        dict: A dictionary containing the probabilities for each label.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If labels are not provided.
    """
    model = ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14-336px')
    processor = ChineseCLIPProcessor.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14-336px')
    
    # Load image
    try:
        image = Image.open(image_path)
    except IOError:
        raise FileNotFoundError(f'Image file not found at {image_path}')

    # Check if labels are provided
    if not labels:
        raise ValueError('Labels must be provided for classification')

    # Preprocess inputs
    inputs = processor(images=image, text=labels, return_tensors="pt", padding=True)

    # Get model outputs
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    
    # Convert probabilities to dictionary
    probs_dict = {label: prob.item() for label, prob in zip(labels, probs.squeeze())}
    
    return probs_dict

# test_function_code --------------------

def test_classify_image_content():
    print("Testing started.")
    
    # Define test image path and labels
    image_path = 'test_image.jpg'
    labels = ['safe', 'explicit', 'violent']

    # Test case 1: Image file exists and labels provided
    print("Testing case [1/3] started.")
    result = classify_image_content(image_path, labels)
    assert isinstance(result, dict), "Test case [1/3] failed: Result should be a dictionary."

    # Test case 2: Image file does not exist
    print("Testing case [2/3] started.")
    try:
        classify_image_content('non_existent.jpg', labels)
        assert False, "Test case [2/3] failed: Should raise FileNotFoundError."
    except FileNotFoundError:
        pass

    # Test case 3: Labels not provided
    print("Testing case [3/3] started.")
    try:
        classify_image_content(image_path, [])
        assert False, "Test case [3/3] failed: Should raise ValueError."
    except ValueError:
        pass
    
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_image_content()