# requirements_file --------------------

!pip install -U transformers>=4.0.0

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def filter_offensive_images(image_path):
    """
    Classify an image and filter out adult or offensive content.

    Parameters:
    image_path (str): The path to the image to be classified.

    Returns:
    bool: True if the image is safe for work, False if it is adult content or offensive.
    """
    image_classifier = pipeline('zero-shot-classification', model='laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
    class_names = ['safe for work', 'adult content', 'offensive']
    result = image_classifier(image=image_path, class_names=class_names)

    # Assume the top result is the right classification
    top_result = result['labels'][0].lower()
    return top_result == 'safe for work'

# test_function_code --------------------

def test_filter_offensive_images():
    print("Testing filter_offensive_images function.")
    # Image paths for test, replace with real image paths or URLs
    safe_image_path = 'path/to/safe_image.jpg'
    adult_image_path = 'path/to/adult_image.jpg'
    offensive_image_path = 'path/to/offensive_image.jpg'

    assert filter_offensive_images(safe_image_path) == True, "Safe image wrongly classified."
    assert filter_offensive_images(adult_image_path) == False, "Adult content not filtered out."
    assert filter_offensive_images(offensive_image_path) == False, "Offensive content not filtered out."
    print("All test cases pass for filter_offensive_images function.")

# Running the test function
test_filter_offensive_images()