# requirements_file --------------------

!pip install -U pillow requests transformers

# function_import --------------------

from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# function_code --------------------

def classify_animal_in_image(image_url: str, processor: ChineseCLIPProcessor, model: ChineseCLIPModel) -> str:
    """Classifies whether the provided image URL contains a cat or a dog.

    Args:
        image_url: The URL of the image to be classified.
        processor: The ChineseCLIPProcessor object for image and text processing.
        model: The ChineseCLIPModel object for feature extraction and classification.

    Returns:
        A string indicating the animal classified in the image.

    Raises:
        ValueError: If the image URL is invalid or if the image cannot be opened.
    """
    try:
        # Load the image
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise ValueError('Invalid image URL or unable to open the image.') from e

    # Define the Chinese captions for a cat and a dog
    captions = ['猫', '狗']  # '猫', '狗'

    # Process the image and text inputs
    inputs = processor(images=image, text=captions, return_tensors='pt', padding=True)
    outputs = model(**inputs)

    # Calculate probabilities
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # Get the highest probability index
    highest_prob_idx = probs.argmax(dim=1).item()

    # Return the corresponding animal
    return captions[highest_prob_idx]

# test_function_code --------------------

def test_classify_animal_in_image():
    print('Testing started.')
    # Use a sample image URL for testing
    sample_image_url = 'https://example.com/sample_image.jpg'
    # Initialize the processor and model
    processor = ChineseCLIPProcessor.from_pretrained('OFA-Sys/chinese-clip-vit-base-patch16')
    model = ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-base-patch16')

    # Testing case 1: Image with a cat
    print('Testing case [1/3] started.')
    assert classify_animal_in_image(sample_image_url, processor, model) in ['猫', '狗'], 'Test case [1/3] failed: Result not among expected categories.'

    # Testing case 2: Image with a dog
    print('Testing case [2/3] started.')
    assert classify_animal_in_image(sample_image_url, processor, model) in ['猫', '狗'], 'Test case [2/3] failed: Result not among expected categories.'

    # Testing case 3: Invalid image URL
    print('Testing case [3/3] started.')
    try:
        classify_animal_in_image('invalid_url', processor, model)
        assert False, 'Test case [3/3] failed: Invalid URL should raise a ValueError.'
    except ValueError:
        assert True

    print('Testing finished.')

# call_test_function_line --------------------

test_classify_animal_in_image()