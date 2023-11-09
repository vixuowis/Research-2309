# function_import --------------------

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def classify_pet_images(image_path: str):
    """
    Classify images of pets into different categories using the pre-trained CLIP model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        dict: A dictionary where the keys are the categories ('a photo of a cat', 'a photo of a dog') and the values are the corresponding classification probabilities.
    """
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    image = Image.open(image_path)
    inputs = processor(text=['a photo of a cat', 'a photo of a dog'], images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return {'a photo of a cat': probs[0].item(), 'a photo of a dog': probs[1].item()}

# test_function_code --------------------

def test_classify_pet_images():
    """
    Test the classify_pet_images function.
    """
    image_path = 'test_image.jpg'  # replace with the path to your test image
    result = classify_pet_images(image_path)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'a photo of a cat', 'a photo of a dog'}
    assert all(isinstance(value, float) for value in result.values())
    assert sum(result.values()) - 1.0 < 1e-6  # the probabilities should sum up to 1

# call_test_function_code --------------------

test_classify_pet_images()