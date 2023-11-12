# function_import --------------------

from transformers import DetrForObjectDetection, DetrImageProcessor
import torch
from PIL import Image

# function_code --------------------

def detect_tables_in_image(image_path: str, threshold: float = 0.9):
    """
    Detect tables in a given image using the DetrForObjectDetection model.

    Args:
        image_path (str): The path to the image file.
        threshold (float, optional): The confidence threshold for detection. Defaults to 0.9.

    Returns:
        list: A list of detected tables information including the label, confidence score, and location.

    Raises:
        PIL.UnidentifiedImageError: If the image file cannot be identified.
    """
    image = Image.open(image_path)
    processor = DetrImageProcessor.from_pretrained('TahaDouaji/detr-doc-table-detection')
    model = DetrForObjectDetection.from_pretrained('TahaDouaji/detr-doc-table-detection')

    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

    detected_tables = []
    for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
        box = [round(i, 2) for i in box.tolist()]
        detected_tables.append({
            'label': model.config.id2label[label.item()],
            'confidence': round(score.item(), 3),
            'location': box
        })

    return detected_tables

# test_function_code --------------------

def test_detect_tables_in_image():
    """Test the detect_tables_in_image function."""
    test_image_path = 'https://placekitten.com/200/300'
    detected_tables = detect_tables_in_image(test_image_path)
    assert isinstance(detected_tables, list)
    for table in detected_tables:
        assert 'label' in table
        assert 'confidence' in table
        assert 'location' in table
        assert isinstance(table['label'], str)
        assert isinstance(table['confidence'], float)
        assert isinstance(table['location'], list)
    print('All Tests Passed')

# call_test_function_code --------------------

test_detect_tables_in_image()