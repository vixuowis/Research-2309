# requirements_file --------------------

!pip install -U Pillow transformers

# function_import --------------------

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def classify_vehicle_image(image_path):
    """
    Classifies an image of a vehicle into categories like car, motorcycle, truck, or bicycle.
    
    Parameters:
    - image_path (str): The file path to the vehicle image to classify.
    
    Returns:
    - str: The classification category (e.g., 'a car', 'a motorcycle', 'a truck', 'a bicycle').
    - float: The probability associated with the classification.
    """
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    image = Image.open(image_path)
    inputs = processor(text=['a car', 'a motorcycle', 'a truck', 'a bicycle'], images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]
    categories = ['a car', 'a motorcycle', 'a truck', 'a bicycle']
    max_idx = probs.argmax()
    return categories[max_idx], probs[max_idx]

# test_function_code --------------------

def test_classify_vehicle_image():
    print("Testing started.")
    test_image_paths = {'car': 'path/to/car_image.jpg','motorcycle': 'path/to/motorcycle_image.jpg','truck': 'path/to/truck_image.jpg','bicycle': 'path/to/bicycle_image.jpg'}
    for vehicle_type, image_path in test_image_paths.items():
        print(f"Testing case for {vehicle_type} started.")
        classification, probability = classify_vehicle_image(image_path)
        assert classification == f"a {vehicle_type}", f"Test case for {vehicle_type} failed: classification '{classification}' does not match expected '{vehicle_type}'."
        assert probability >= 0.0 and probability <= 1.0, f"Test case for {vehicle_type} failed: probability '{probability}' out of range [0, 1]."
        print(f"Testing case for {vehicle_type} passed.")
    print("Testing finished.")