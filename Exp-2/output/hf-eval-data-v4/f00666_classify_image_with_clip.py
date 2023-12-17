# requirements_file --------------------

!pip install -U PIL transformers

# function_import --------------------

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def classify_image_with_clip(image_path, categories):
    """
    Classify an image into one of the specified categories using the CLIP model.

    :param image_path: str, the path to the image file.
    :param categories: list of str, the categories to classify the image into.
    :return: dict, the probabilities for each category.
    """
    # Load the pre-trained CLIP model and processor
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')

    # Load the image
    image = Image.open(image_path)

    # Process the image and the categories
    inputs = processor(text=categories, images=image, return_tensors="pt", padding=True)

    # Get the model outputs
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image

    # Calculate the probabilities
    probs = logits_per_image.softmax(dim=1)

    # Convert probabilities to a dictionary
    category_probs = {category: prob.item() for category, prob in zip(categories, probs.squeeze())}
    return category_probs

# test_function_code --------------------

def test_classify_image_with_clip():
    print("Testing classify_image_with_clip function.")
    categories = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
    test_image_path = 'test_image.jpg'  # A test image path

    # Test case 1: Classify an image
    print("Running test case 1")
    result = classify_image_with_clip(test_image_path, categories)
    assert isinstance(result, dict), "The result should be a dictionary."
    assert all(category in result for category in categories), "The result should have all the categories."
    assert all(isinstance(value, float) for value in result.values()), "All category probabilities should be float values."
    print("Test case 1 passed.")

    print("Testing finished.")

# Execute the test function
test_classify_image_with_clip()