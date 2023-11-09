# function_import --------------------

from transformers import pipeline
from PIL import Image

# function_code --------------------

def detect_ai_generated_anime(image_path):
    """
    This function uses a pre-trained model to classify whether the provided anime art is created by a human or generated through AI.

    Args:
        image_path (str): The path to the image file to be classified.

    Returns:
        str: The classification result, indicating whether the provided image is created by a human or AI-generated.
    """
    # Load the image from the provided path
    image = Image.open(image_path)

    # Create an image classification model with the pre-trained model
    anime_detector = pipeline('image-classification', model='saltacc/anime-ai-detect')

    # Pass the image to the image classification model
    classification_result = anime_detector(image)

    # Return the classification result
    return classification_result

# test_function_code --------------------

def test_detect_ai_generated_anime():
    """
    This function tests the 'detect_ai_generated_anime' function by using a sample image.
    """
    # Define the path to the sample image
    sample_image_path = 'path_to_sample_image.jpg'

    # Call the 'detect_ai_generated_anime' function with the sample image
    result = detect_ai_generated_anime(sample_image_path)

    # Assert that the result is not None (i.e., the function should always return a result)
    assert result is not None

# call_test_function_code --------------------

test_detect_ai_generated_anime()