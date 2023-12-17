# requirements_file --------------------

!pip install -U transformers pillow

# function_import --------------------

from transformers import pipeline
from PIL import Image

# function_code --------------------

def determine_art_origin(image_path):
    """
    Determine whether an anime art image is created by a human artist or AI-generated.

    Args:
        image_path (str): The file path to the anime art image.

    Returns:
        dict: The classification result indicating the origin of the art.
    """
    # Load the user-submitted image
    image = Image.open(image_path)

    # Initialize the classification pipeline with the pre-trained model
    anime_detector = pipeline('image-classification', model='saltacc/anime-ai-detect')

    # Perform classification to determine the origin
    classification_result = anime_detector(image)
    return classification_result

# test_function_code --------------------

def test_determine_art_origin():
    print("Testing started.")

    # Test case 1: Check with a known human-created art
    human_art_path = 'human_art_sample.jpg'  # Replace with actual file path
    result_human = determine_art_origin(human_art_path)
    assert result_human[0]['label'] == 'Human', f"Test case failed: Expected Human, but got {result_human[0]['label']}"

    # Test case 2: Check with a known AI-generated art
    ai_art_path = 'ai_art_sample.jpg'  # Replace with actual file path
    result_ai = determine_art_origin(ai_art_path)
    assert result_ai[0]['label'] == 'AI', f"Test case failed: Expected AI, but got {result_ai[0]['label']}"
    
    print("Testing finished.")

# Run the test function
if __name__ == '__main__':
    test_determine_art_origin()