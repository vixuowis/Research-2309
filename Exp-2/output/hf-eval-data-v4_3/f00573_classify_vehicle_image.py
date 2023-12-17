# requirements_file --------------------

import subprocess

requirements = ["PIL", "transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# function_code --------------------

def classify_vehicle_image(image_path):
    """
    Classify an image of a vehicle into categories: car, motorcycle, truck, or bicycle.

    Args:
        image_path (str): The path to the image file to classify.

    Returns:
        dict: A dictionary containing the class probabilities.

    Raises:
        FileNotFoundError: If the image_path does not exist.
    """
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        raise FileNotFoundError(f'The image file {image_path} was not found.')

    text_labels = ['a car', 'a motorcycle', 'a truck', 'a bicycle']
    inputs = processor(text=text_labels, images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).tolist()[0]

    return {label: prob for label, prob in zip(text_labels, probs)}

# test_function_code --------------------

def test_classify_vehicle_image():
    print("Testing started.")
    test_image_path = 'test_vehicle.jpg'

    # Prepare test case
    text_labels = ['a car', 'a motorcycle', 'a truck', 'a bicycle']

    # Testing case 1: Test with a simple example
    print("Testing case [1/1] started.")
    result = classify_vehicle_image(test_image_path)
    assert len(result) == 4 and all(label in result for label in text_labels), f"Test case [1/1] failed: Expected labels not found in result: {result}"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_vehicle_image()