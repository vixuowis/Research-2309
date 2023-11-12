# function_import --------------------

from transformers import DeformableDetrForObjectDetection, AutoImageProcessor
from PIL import Image

# function_code --------------------

def detect_objects(image_path):
    """
    Detect objects in an image using the Deformable DETR model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        dict: The outputs from the Deformable DETR model.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    image = Image.open(image_path)
    processor = AutoImageProcessor.from_pretrained('SenseTime/deformable-detr')
    model = DeformableDetrForObjectDetection.from_pretrained('SenseTime/deformable-detr')
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    return outputs

# test_function_code --------------------

def test_detect_objects():
    """
    Test the detect_objects function.
    """
    # Test with a valid image file
    outputs = detect_objects('test_image.jpg')
    assert isinstance(outputs, dict)
    # Test with a non-existent image file
    try:
        detect_objects('non_existent_image.jpg')
    except FileNotFoundError:
        pass
    else:
        assert False, 'Expected a FileNotFoundError.'
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_detect_objects())