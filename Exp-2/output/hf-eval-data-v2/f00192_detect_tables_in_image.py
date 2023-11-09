# function_import --------------------

from transformers import DetrForObjectDetection, DetrImageProcessor
import torch
from PIL import Image
import requests

# function_code --------------------

def detect_tables_in_image(image_path, model_name='TahaDouaji/detr-doc-table-detection', threshold=0.9):
    """
    Detects tables in a given image using the specified model.

    Args:
        image_path (str): The path to the image file.
        model_name (str, optional): The name of the model to use for detection. Defaults to 'TahaDouaji/detr-doc-table-detection'.
        threshold (float, optional): The confidence threshold for detection. Defaults to 0.9.

    Returns:
        list: A list of detected tables, each represented as a dictionary with 'label', 'confidence', and 'location' keys.
    """
    image = Image.open(image_path)
    processor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name)

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
    """
    Tests the detect_tables_in_image function.
    """
    image_url = 'https://example.com/test_image.jpg'
    image_path = 'test_image.jpg'
    response = requests.get(image_url)
    with open(image_path, 'wb') as f:
        f.write(response.content)

    detected_tables = detect_tables_in_image(image_path)
    assert isinstance(detected_tables, list)
    for table in detected_tables:
        assert 'label' in table
        assert 'confidence' in table
        assert 'location' in table
        assert isinstance(table['label'], str)
        assert isinstance(table['confidence'], float)
        assert isinstance(table['location'], list)
        assert len(table['location']) == 4
        assert all(isinstance(i, float) for i in table['location'])

# call_test_function_code --------------------

test_detect_tables_in_image()