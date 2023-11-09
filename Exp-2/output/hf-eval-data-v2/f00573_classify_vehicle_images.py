# function_import --------------------

from transformers import CLIPModel, CLIPProcessor
from PIL import Image

# function_code --------------------

def classify_vehicle_images(image_path):
    """
    Classify images of vehicles including cars, motorcycles, trucks, and bicycles, based on their appearance.

    Args:
        image_path (str): The path to the image file.

    Returns:
        dict: A dictionary where keys are vehicle types ('a car', 'a motorcycle', 'a truck', 'a bicycle') and values are their corresponding probabilities.
    """
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    image = Image.open(image_path)
    inputs = processor(text=['a car', 'a motorcycle', 'a truck', 'a bicycle'], images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return {vehicle: prob for vehicle, prob in zip(['a car', 'a motorcycle', 'a truck', 'a bicycle'], probs.tolist())}

# test_function_code --------------------

def test_classify_vehicle_images():
    """
    Test the function classify_vehicle_images.
    """
    image_path = 'test_image.jpg'  # replace with your test image path
    result = classify_vehicle_images(image_path)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert set(result.keys()) == set(['a car', 'a motorcycle', 'a truck', 'a bicycle']), 'The keys of the result should be vehicle types.'
    assert all(0 <= prob <= 1 for prob in result.values()), 'The probabilities should be between 0 and 1.'

# call_test_function_code --------------------

test_classify_vehicle_images()