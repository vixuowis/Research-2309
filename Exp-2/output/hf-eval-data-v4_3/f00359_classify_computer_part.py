# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch", "Pillow"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image


# function_code --------------------

def classify_computer_part(image_path):
    """
    Classifies a computer part in an uploaded image.

    Args:
        image_path (str): The file path to the user uploaded image.

    Returns:
        str: The predicted label for the computer part in the image.

    Raises:
        FileNotFoundError: If the image_path does not exist.
        ValueError: If the image cannot be processed.
    """
    try:
        # Load and preprocess the image
        image = Image.open(image_path)
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        inputs = processor(images=image, return_tensors='pt')

        # Load the pre-trained Vision Transformer model
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

        # Get model predictions
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

        # Retrieve the human-readable label
        predicted_label = model.config.id2label[predicted_class_idx]
        return predicted_label
    except FileNotFoundError:
        raise FileNotFoundError(f'Image file not found: {image_path}')
    except Exception as e:
        raise ValueError(f'Image processing failed: {e}')


# test_function_code --------------------

def test_classify_computer_part():
    print("Testing started.")

    # Test case 1: Valid image file
    print("Testing case [1/2] started.")
    predicted_label = classify_computer_part('valid_image_file_path')
    assert predicted_label is not None, f"Test case [1/2] failed: Expected a label, got {predicted_label}"

    # Test case 2: Invalid image file
    print("Testing case [2/2] started.")
    try:
        classify_computer_part('invalid_image_file_path')
        assert False, "Test case [2/2] failed: Expected FileNotFoundError"
    except FileNotFoundError:
        pass

    print("Testing finished.")


# call_test_function_line --------------------

test_classify_computer_part()