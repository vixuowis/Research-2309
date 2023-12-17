# requirements_file --------------------

!pip install -U transformers Pillow

# function_import --------------------

from transformers import DeformableDetrForObjectDetection, AutoImageProcessor
from PIL import Image

# function_code --------------------

def detect_objects_in_image(image_path):
    """
    Detect objects in an image using the Deformable DETR model.

    Parameters:
        image_path (str): The path to the image file where objects need to be detected.

    Returns:
        dict: The object detection results containing the detected objects and their details.
    """
    processor = AutoImageProcessor.from_pretrained('SenseTime/deformable-detr')
    model = DeformableDetrForObjectDetection.from_pretrained('SenseTime/deformable-detr')
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    return outputs

# test_function_code --------------------

def test_detect_objects_in_image():
    print("Testing detect_objects_in_image started.")
    test_image_path = 'test_image.jpg'  # replace with a path to a test image
    result = detect_objects_in_image(test_image_path)
    print("Testing detect_objects_in_image finished.")
    assert type(result) == dict, f"Test failed: The output should be a dict, got {type(result)} instead."

# Run the test function
test_detect_objects_in_image()