# requirements_file --------------------

import subprocess

requirements = ["Pillow", "transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def classify_image(image_path, categories):
    """
    Classify an image into categories using CLIP model.

    Args:
        image_path (str): The file path to the image to classify.
        categories (list): A list of category descriptions.

    Returns:
        dict: A dictionary with categories as keys and their respective probabilities as values.

    Raises:
        FileNotFoundError: If the image_path does not exist.
        ValueError: If categories list is empty.
    """
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')
    image = Image.open(image_path)

    inputs = processor(text=categories, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # Convert the probabilities to a dictionary
    category_probs = {category: float(prob) for category, prob in zip(categories, probs[0])}
    return category_probs

# test_function_code --------------------

def test_classify_image():
    print("Testing started.")
    image_path = 'sample_image.jpg'
    categories = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
    # Assuming we have sample image and categories

    # Test case 1: Valid image and categories
    print("Testing case [1/1] started.")
    probs = classify_image(image_path, categories)
    assert isinstance(probs, dict), f"Test case [1/1] failed: Expected dict, got {type(probs)}"
    assert all(category in probs for category in categories), f"Test case [1/1] failed: Missing categories in the results."
    print("Testing finished.")

# Run the test function
test_classify_image()

# call_test_function_line --------------------

test_classify_image()