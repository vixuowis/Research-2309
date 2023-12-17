# requirements_file --------------------

import subprocess

requirements = ["transformers", "PIL"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image

# function_code --------------------

def segment_image(image_path):
    """
    Segments an image using the pre-trained 'nvidia/segformer-b5-finetuned-cityscapes-1024-1024' model.

    Args:
        image_path (str): The file path to the image that needs to be segmented.

    Returns:
        Tensor: A tensor containing the logits of the segmentation outputs.

    Raises:
        FileNotFoundError: If the image_path does not exist.
        ValueError: If the image_path is not an image file.
    """
    feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b5-finetuned-cityscapes-1024-1024')
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-cityscapes-1024-1024')
    
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        raise FileNotFoundError(f'The image at {image_path} does not exist.')
    except (IOError, ValueError):
        raise ValueError(f'The file at {image_path} is not a valid image.')
    
    inputs = feature_extractor(images=image, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    return logits

# test_function_code --------------------

def test_segment_image():
    print("Testing started.")
    test_image = 'test_image.jpg'  # Use an actual image file for testing

    # Testing case 1: Image Exists
    print("Testing case [1/3] started.")
    try:
        _ = segment_image(test_image)
        print("Test case [1/3] finished successfully.")
    except Exception as e:
        print(f"Test case [1/3] failed: {e}")

    # Testing case 2: Image Does Not Exist
    print("Testing case [2/3] started.")
    try:
        _ = segment_image('non_existent_image.jpg')
        print("Test case [2/3] failed: Expected FileNotFoundError not thrown.")
    except FileNotFoundError:
        print("Test case [2/3] finished successfully.")

    # Testing case 3: Invalid Image Path
    print("Testing case [3/3] started.")
    try:
        _ = segment_image('invalid_image_path.txt')
        print("Test case [3/3] failed: Expected ValueError not thrown.")
    except ValueError:
        print("Test case [3/3] finished successfully.")
    
    print("Testing finished.")

# call_test_function_line --------------------

test_segment_image()