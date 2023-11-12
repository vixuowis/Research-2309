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
        dict: A dictionary with vehicle types as keys and their corresponding probabilities as values.

    Raises:
        OSError: If the image file cannot be opened.
    """
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    image = Image.open(image_path)
    inputs = processor(text=['a car', 'a motorcycle', 'a truck', 'a bicycle'], images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return {'car': probs[0].item(), 'motorcycle': probs[1].item(), 'truck': probs[2].item(), 'bicycle': probs[3].item()}

# test_function_code --------------------

def test_classify_vehicle_images():
    """
    Test the function classify_vehicle_images.
    """
    # Test with a car image
    result = classify_vehicle_images('car.jpg')
    assert isinstance(result, dict)
    assert set(result.keys()) == {'car', 'motorcycle', 'truck', 'bicycle'}
    assert all(isinstance(value, float) for value in result.values())

    # Test with a motorcycle image
    result = classify_vehicle_images('motorcycle.jpg')
    assert isinstance(result, dict)
    assert set(result.keys()) == {'car', 'motorcycle', 'truck', 'bicycle'}
    assert all(isinstance(value, float) for value in result.values())

    # Test with a truck image
    result = classify_vehicle_images('truck.jpg')
    assert isinstance(result, dict)
    assert set(result.keys()) == {'car', 'motorcycle', 'truck', 'bicycle'}
    assert all(isinstance(value, float) for value in result.values())

    # Test with a bicycle image
    result = classify_vehicle_images('bicycle.jpg')
    assert isinstance(result, dict)
    assert set(result.keys()) == {'car', 'motorcycle', 'truck', 'bicycle'}
    assert all(isinstance(value, float) for value in result.values())

    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_vehicle_images()