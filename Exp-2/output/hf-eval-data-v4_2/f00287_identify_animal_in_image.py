# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def identify_animal_in_image(image_path: str) -> dict:
    """
    Identifies whether the animal in the provided image is a cat or a dog using
    a pre-trained image classification model.

    Args:
        image_path: The path to the image file to be classified.

    Returns:
        A dictionary containing the predicted class and confidence score.

    Raises:
        FileNotFoundError: If the image file does not exist at the provided path.
        Exception: If the classification model encounters an error during processing.
    """
    try:
        image_classifier = pipeline('image-classification', model='laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft')
        result = image_classifier(image_path, ['cat', 'dog'])
        return result[0]
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Image file not found: {image_path}") from e
    except Exception as e:
        raise Exception("Error during image classification") from e


# test_function_code --------------------


# Define the test function for identify_animal_in_image
import os

def test_identify_animal_in_image():
    print("Testing started.")
    # Assuming test images are located in a directory named 'test_images'
    test_images_dir = 'test_images'
    test_images = os.listdir(test_images_dir)

    if not test_images:
        raise Exception("No test images found")

    # Run test cases for each test image
    for i, image_name in enumerate(test_images, 1):
        image_path = os.path.join(test_images_dir, image_name)
        print(f"Testing case [{i}/{len(test_images)}] started.")
        result = identify_animal_in_image(image_path)
        assert result["label"] in ["cat", "dog"], f"Test case [{i}/{len(test_images)}] failed: Unexpected label." 

    print("Testing finished.")


# call_test_function_line --------------------

test_identify_animal_in_image()