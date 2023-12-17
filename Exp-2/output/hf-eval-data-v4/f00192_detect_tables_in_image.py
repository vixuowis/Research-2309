# requirements_file --------------------

!pip install -U transformers torch PIL

# function_import --------------------

from transformers import DetrForObjectDetection, DetrImageProcessor
import torch
from PIL import Image

# function_code --------------------

def detect_tables_in_image(image_path):
    # Load the image
    image = Image.open(image_path)

    # Initialize the image processor and the model
    processor = DetrImageProcessor.from_pretrained('TahaDouaji/detr-doc-table-detection')
    model = DetrForObjectDetection.from_pretrained('TahaDouaji/detr-doc-table-detection')

    # Process the image
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)

    # Calculate the target sizes using the image size
    target_sizes = torch.tensor([image.size[::-1]])

    # Process the outputs with the object detection processor
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Extract detected tables information
    detected_tables = []
    for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
        label_name = model.config.id2label[label.item()]
        confidence = round(score.item(), 3)
        bbox = [round(i, 2) for i in box.tolist()]
        detected_tables.append({'label': label_name, 'confidence': confidence, 'box': bbox})

    return detected_tables

# test_function_code --------------------

def test_detect_tables_in_image():
    print("Testing started.")
    # Load a sample test image path
    sample_image_path = 'test_image.jpg'  # Placeholder for image path

    # Test case
    print("Testing case started.")
    detected_tables = detect_tables_in_image(sample_image_path)
    assert len(detected_tables) > 0, f"No tables detected, test failed."

    # Test passed
    print("Test passed. Detected tables: ", detected_tables)

# Run the test function
test_detect_tables_in_image()