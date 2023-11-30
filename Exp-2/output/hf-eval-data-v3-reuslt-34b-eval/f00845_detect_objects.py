# function_import --------------------

import requests
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# function_code --------------------

def detect_objects(url: str, texts: list, model_name: str = 'google/owlvit-large-patch14', score_threshold: float = 0.1):
    """
    Detect objects in an image based on specific text phrases using the OwlViT model.

    Args:
        url (str): The URL of the image.
        texts (list): A list of text descriptions.
        model_name (str, optional): The name of the OwlViT model. Defaults to 'google/owlvit-large-patch14'.
        score_threshold (float, optional): The score threshold for filtering detections. Defaults to 0.1.

    Returns:
        None. Prints the detected objects, their confidence scores, and bounding box locations.
    """
    
    # Load OwlViT model
    processor = OwlViTProcessor.from_pretrained(model_name)
    model = OwlViTForObjectDetection.from_pretrained(model_name)
    
    # Create batch of inputs to process
    with urllib.request.urlopen(url) as image_file:
        input_data = [{'url': url, 'texts': texts}]
        
    # Preprocess inputs
    processed_input = processor(images=image_file, args=input_data)
    
    # Run through model and post-process outputs to get object detections
    with torch.no_grad():
        outputs = model(**processed_input)
        
    # Print object detection results
    print(f"Detected objects (with text phrases):")
    for box in outputs.object_boxes[0]:
        if outputs.object_confidences[0][outputs.object_labels[0].index(max(outputs.object_labels[0]))] >= score_threshold:
            print(f"{processor.tokenizer.decode(outputs.text_seq[0])} - Confidence: {round(100 * outputs.object_confidences[0][outputs.object_labels[0].index(max(outputs.object_labels[0]))], 2)}%", end=' ')
            print({'x': round(box[0]), 'y': round(box[1]), 'width': round(box[2] - box[0]), 'height': round(box[3] - box[1])})


# test_function_code --------------------

def test_detect_objects():
    """
    Test the detect_objects function.
    """
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    texts = ['a photo of a cat', 'a photo of a dog']
    try:
        detect_objects(url, texts)
        print('Test passed.')
    except Exception as e:
        print('Test failed. Error: ', e)


# call_test_function_code --------------------

test_detect_objects()