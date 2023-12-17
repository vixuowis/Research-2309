# requirements_file --------------------

!pip install -U transformers PIL requests

# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def identify_street_picture_location(image_url, choices):
    """
    Identify the location where a street-level picture was taken.

    Args:
        image_url (str): URL of the street-level image.
        choices (list): A list of location names to consider as possible places.

    Returns:
        str: The name of the location with the highest probability.
    """
    # Load the pre-trained model and processor
    model = CLIPModel.from_pretrained('geolocal/StreetCLIP')
    processor = CLIPProcessor.from_pretrained('geolocal/StreetCLIP')
    
    # Load the image from the provided URL
    image = Image.open(requests.get(image_url, stream=True).raw)
    
    # Process the text labels list and image into tensors
    inputs = processor(text=choices, images=image, return_tensors='pt', padding=True)
    
    # Execute the model on the inputs to get logits per image
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    
    # Convert the logits to probabilities
    probs = logits_per_image.softmax(dim=1)
    
    # Identify the location with the highest probability
    max_prob_idx = probs.argmax()
    
    return choices[max_prob_idx]

# test_function_code --------------------

def test_identify_street_picture_location():
    print("Testing started.")
    # This is a hypothetical test case since the actual URLs and choices
    # will depend on the user-provided data.
    # Test case 1
    image_url = 'https://example.com/test-image-1.jpg'
    choices = ['Paris', 'New York', 'Tokyo']
    expected_output = 'New York'  # Assuming 'New York' is the expected location
    output = identify_street_picture_location(image_url, choices)
    assert expected_output == output, f"Test case failed: Expected {expected_output}, got {output}"
    
    print("Testing finished.")

test_identify_street_picture_location()