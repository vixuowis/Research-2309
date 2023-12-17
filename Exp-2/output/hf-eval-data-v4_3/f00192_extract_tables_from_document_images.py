# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch", "Pillow", "requests"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import DetrForObjectDetection, DetrImageProcessor
import torch
from PIL import Image

# function_code --------------------

def extract_tables_from_document_images(image_path):
    """
    Detect tables in scanned document images using DETR model.

    Args:
        image_path (str): The path to the document image file.

    Returns:
        List[Tuple[str, float, List[float]]]: A list of tuples containing label, confidence score,
                                           and bounding box coordinates for each detected table.

    Raises:
        FileNotFoundError: If the image file does not exist at the provided path.
        Exception: If the model fails to process the image or output detection results.
    """
    image = Image.open(image_path)
    processor = DetrImageProcessor.from_pretrained('TahaDouaji/detr-doc-table-detection')
    model = DetrForObjectDetection.from_pretrained('TahaDouaji/detr-doc-table-detection')

    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detections = []
    for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
        if score > 0.9:
            box = [round(i, 2) for i in box.tolist()]
            label_desc = model.config.id2label[label.item()]
            detections.append((label_desc, round(score.item(), 3), box))

    return detections

# test_function_code --------------------

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
def test_extract_tables_from_document_images():
    print("Testing started.")
    # Assuming 'sample-image.jpg' is a valid image in the test dataset
    sample_image_path = 'sample-image.jpg'

    # Test case 1: Testing with a valid image path
    print("Testing case [1/2] started.")
    detection_results = extract_tables_from_document_images(sample_image_path)
    assert isinstance(detection_results, list), "Test case [1/2] failed: The result should be a list."
    assert all(isinstance(item, tuple) and len(item) == 3 for item in detection_results), "Test case [1/2] failed: Each item in the result list should be a tuple of length 3."

    # Test case 2: Testing with a non-existent image path
    print("Testing case [2/2] started.")
    try:
        extract_tables_from_document_images('non-existent-image.jpg')
        assert False, "Test case [2/2] failed: FileNotFoundError expected."
    except FileNotFoundError:
        pass  # Expected exception
    except Exception as e:
        assert False, f"Test case [2/2] failed: Unexpected exception {e}."
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_tables_from_document_images()