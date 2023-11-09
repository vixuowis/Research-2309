from transformers import DetrForObjectDetection, DetrImageProcessor
import torch
from PIL import Image


def detect_tables_in_image(image_path, model_name='TahaDouaji/detr-doc-table-detection', threshold=0.9):
    """
    This function detects tables in a given image using the DetrForObjectDetection model.
    
    Parameters:
    image_path (str): The path to the image file.
    model_name (str): The name of the pretrained model. Default is 'TahaDouaji/detr-doc-table-detection'.
    threshold (float): The confidence threshold for detection. Default is 0.9.
    
    Returns:
    list: A list of detected tables with their confidence scores and locations.
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
        detected_tables.append((model.config.id2label[label.item()], round(score.item(), 3), box))

    return detected_tables