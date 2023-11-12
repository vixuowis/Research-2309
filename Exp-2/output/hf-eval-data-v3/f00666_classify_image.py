# function_import --------------------

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# function_code --------------------

def classify_image(image_path: str, categories: list) -> torch.Tensor:
    """
    Classify an image into given categories using a pre-trained CLIP model.

    Args:
        image_path (str): The path to the image file.
        categories (list): The list of categories to classify the image into.

    Returns:
        torch.Tensor: The probabilities of the image belonging to each category.

    Raises:
        PIL.UnidentifiedImageError: If the image file cannot be identified.
    """
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')

    image = Image.open(image_path)

    inputs = processor(text=categories, images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    return probs

# test_function_code --------------------

def test_classify_image():
    """Test the classify_image function."""
    image_path = 'https://placekitten.com/200/300'
    categories = ['a photo of a cat', 'a photo of a dog', 'a photo of a bird']
    probs = classify_image(image_path, categories)
    assert isinstance(probs, torch.Tensor), 'The return type should be a torch.Tensor.'
    assert probs.shape[0] == len(categories), 'The number of probabilities should be equal to the number of categories.'
    assert torch.all(probs >= 0) and torch.all(probs <= 1), 'All probabilities should be between 0 and 1.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_image()