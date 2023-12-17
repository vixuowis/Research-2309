# requirements_file --------------------

!pip install -U PIL requests transformers

# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def classify_image_with_clip(image_url, labels):
    """
    Classify the content of the provided image URL using a pre-trained CLIP model.

    Args:
    image_url (str): URL of the image to classify.
    labels (list of str): A list of labels for classification.

    Returns:
    dict: Dictionary containing labels and their corresponding probabilities.
    """
    # Load the pre-trained CLIP model and processor
    model = CLIPModel.from_pretrained('flax-community/clip-rsicd-v2')
    processor = CLIPProcessor.from_pretrained('flax-community/clip-rsicd-v2')

    # Download the image and open it
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Process the image and the provided labels
    inputs = processor(text=[f'a photo of a {l}' for l in labels], images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)

    # Extract logits and apply softmax to get probabilities
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # Return the labels and probabilities as a dictionary
    return {label: prob.item() for label, prob in zip(labels, probs[0])}

# test_function_code --------------------

def test_classify_image_with_clip():
    print("Testing started.")
    image_url = 'https://example.com/test_image.jpg'  # Replace with a valid image URL
    labels = ['residential area', 'playground', 'stadium', 'forest', 'airport']
    expected_labels = set(labels)

    # Test case: Check if the function returns results for all provided labels
    print("Testing case [1/1] started.")
    result = classify_image_with_clip(image_url, labels)
    assert set(result.keys()) == expected_labels, f"Test case [1/1] failed: Missing or unexpected labels in result {result}"
    print("Testing case [1/1] passed.")
    print("Testing finished.")

# Run the test function
test_classify_image_with_clip()