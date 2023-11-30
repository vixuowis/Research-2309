# function_import --------------------

from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# function_code --------------------

def detect_kitchen_objects(image_path: str, score_threshold: float):
    """
    Detects kitchen objects in an image using the OwlViT object detection model.

    Args:
        image_path (str): The path to the image file.
        score_threshold (float): The confidence score threshold for object detection.

    Returns:
        None. Prints the detected objects, their confidence scores, and their locations.

    Raises:
        FileNotFoundError: If the image file does not exist.
        RuntimeError: If there is a problem loading the model or processing the image.
    """
    # Verify that the image exists
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        raise FileNotFoundError("The image file was not found.")
    
    # Load the model and processor
    processor = OwlViTProcessor.from_pretrained("vovk/owlvit-base-1m-sampling")
    model = OwlViTForObjectDetection.from_pretrained("vovk/owlvit-base-1m-sampling", num_labels=20)
    
    # Process the image
    try:
        processed_image = processor(image, return_tensors="pt")
    except RuntimeError as error:
        raise RuntimeError("There was a problem loading the model.") from error
    
    # Perform object detection
    outputs = model(**processed_image)
    
    # Print the detected objects and their locations
    if len(outputs.logits[0]) > 0:
        print("Detected objects:")
        
        labels = outputs.logits[0].softmax(-1)[0]
        scores, classes = labels.detach().topk(labels.shape[-1])
                    
        for score, class_index in zip(scores, classes):
            if score > score_threshold:
                print(f"- {processor.decode_label(int(class_index)).title()}, confidence score: {score}")
    else:
        print("No objects detected.")

# test_function_code --------------------

def test_detect_kitchen_objects():
    """
    Tests the detect_kitchen_objects function.
    """
    try:
        detect_kitchen_objects('test_image.jpg', 0.1)
        print('Test passed')
    except Exception as e:
        print(f'Test failed: {str(e)}')


# call_test_function_code --------------------

test_detect_kitchen_objects()