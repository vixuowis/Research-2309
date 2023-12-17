# requirements_file --------------------

!pip install -U transformers PIL

# function_import --------------------

from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image

# function_code --------------------

def classify_device_image(image_path: str) -> int:
    """
    Classify an image of a device and determine whether it is a cell phone, laptop, or smartwatch.

    Args:
        image_path (str): The file path of the image to classify.

    Returns:
        int: The index of the classified device type (0 for cell phone, 1 for laptop, 2 for smartwatch).

    Raises:
        FileNotFoundError: If the image_path does not exist.
        Exception: If the image could not be processed or classification could not be performed.
    """
    try:
        # Load the pre-trained model and feature extractor
        model = ViTForImageClassification.from_pretrained('lysandre/tiny-vit-random')
        feature_extractor = ViTFeatureExtractor.from_pretrained('lysandre/tiny-vit-random')

        # Open the image file
        image = Image.open(image_path)

        # Preprocess the image
        input_image = feature_extractor(images=image, return_tensors='pt')

        # Classify the image
        output = model(**input_image)
        device_type_index = output.logits.argmax(dim=1).item()

        return device_type_index
    except FileNotFoundError:
        raise FileNotFoundError(f'Image file not found at {image_path}')
    except Exception as e:
        raise Exception(f'Error occurred during classification: {e}')

# test_function_code --------------------

def test_classify_device_image():
    print("Testing started.")
    # Test image paths
    test_images = {
        'cell_phone': 'cell_phone.jpg',
        'laptop': 'laptop.jpg',
        'smartwatch': 'smartwatch.jpg'
    }

    # Expected results for each device type
    expected_results = {'cell_phone': 0, 'laptop': 1, 'smartwatch': 2}

    for i, (device, image_path) in enumerate(test_images.items(), 1):
        print(f"Testing case [{i}/3] started.")
        # Run the classification function
        result = classify_device_image(image_path)
        # Check if the result matches the expected value
        assert result == expected_results[device], f"Test case [{i}/3] failed: Classified as {result}, expected {expected_results[device]}"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_device_image()