# function_import --------------------

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def classify_pet_images(image_path: str, categories: list) -> dict:
    '''
    Classify images of pets into different categories using the pre-trained CLIP model.

    Args:
        image_path (str): The path to the image file.
        categories (list): A list of categories for the pet images (e.g., 'a photo of a cat' or 'a photo of a dog').

    Returns:
        dict: A dictionary where the keys are the categories and the values are the corresponding classification probabilities.
    '''
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    image = Image.open(image_path)
    inputs = processor(text=categories, images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return {category: prob for category, prob in zip(categories, probs.tolist()[0])}

# test_function_code --------------------

def test_classify_pet_images():
    '''
    Test the function classify_pet_images.
    '''
    # Test case 1: Classify an image of a cat
    image_path = 'https://placekitten.com/200/300'
    categories = ['a photo of a cat', 'a photo of a dog']
    result = classify_pet_images(image_path, categories)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(categories)

    # Test case 2: Classify an image of a dog
    image_path = 'https://placedog.net/500'
    categories = ['a photo of a cat', 'a photo of a dog', 'a photo of a bird']
    result = classify_pet_images(image_path, categories)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(categories)

    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_pet_images()