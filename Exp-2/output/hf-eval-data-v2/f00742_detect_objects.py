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
        dict: The detection results.
    """
    # Load the image
    image = Image.open(image_path)

    # Initialize the processor and model
    processor = AutoImageProcessor.from_pretrained('SenseTime/deformable-detr')
    model = DeformableDetrForObjectDetection.from_pretrained('SenseTime/deformable-detr')

    # Process the image and run the model
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)

    return outputs

# test_function_code --------------------

def test_detect_objects():
    """
    Test the detect_objects function.
    """
    # Use a sample image for testing
    image_path = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    # Run the function
    outputs = detect_objects(image_path)

    # Check the output type
    assert isinstance(outputs, dict), 'The output should be a dictionary.'

# call_test_function_code --------------------

test_detect_objects()