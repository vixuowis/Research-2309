# function_import --------------------

from transformers import pipeline
from PIL import Image

# function_code --------------------

def detect_ai_generated_anime(image_path: str) -> str:
    """
    Detects whether the provided anime art is created by a human or AI-generated.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The classification result indicating whether the provided image is created by a human or AI-generated.

    Raises:
        FileNotFoundError: If the image file does not exist at the provided path.
    """
    image = Image.open(image_path)
    anime_detector = pipeline('image-classification', model='saltacc/anime-ai-detect')
    classification_result = anime_detector(image)
    return classification_result

# test_function_code --------------------

def test_detect_ai_generated_anime():
    """
    Tests the detect_ai_generated_anime function with various test cases.
    """
    # Test with a known human-created anime art
    result = detect_ai_generated_anime('path_to_human_created_anime_art.jpg')
    assert result == 'Human', 'Test case 1 failed'

    # Test with a known AI-generated anime art
    result = detect_ai_generated_anime('path_to_ai_generated_anime_art.jpg')
    assert result == 'AI', 'Test case 2 failed'

    # Test with a non-existing file path
    try:
        result = detect_ai_generated_anime('non_existing_file_path.jpg')
    except FileNotFoundError:
        pass
    else:
        assert False, 'Test case 3 failed - Expected a FileNotFoundError'

    print('All tests passed')

# call_test_function_code --------------------

test_detect_ai_generated_anime()