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

    # Load the OwlViT processor used for inference with pretrained weights
    try:
        processor = OwlViTProcessor.from_pretrained("deepmotion/owl-vit")
    except Exception as e:
        raise RuntimeError(f"Failed to load the model due to {e}.")
    
    # Load the OwlViT object detection model with pretrained weights
    try:
        model = OwlViTForObjectDetection.from_pretrained("deepmotion/owl-vit")
    except Exception as e:
        raise RuntimeError(f"Failed to load the model due to {e}.")
        
    # Load the image file for inference
    try: 
        img = Image.open(image_path)
    except FileNotFoundError as e:
        raise FileNotFoundError("The specified image file does not exist.")
    
    # Process the input image to tensor and apply normalization
    try:
        processed_img = processor(images=img, return_tensors="pt") 
        input_tensor = processed_img.pixel_values.unsqueeze(0)
        normalized_tensor = (input_tensor - torch.mean(input_tensor)) / torch.std(input_tensor).item()
    except Exception as e:
        raise RuntimeError(f"Failed to process the image for inference due to {e}.")
        
    # Predict using the OwlViT object detection model
    try: 
        model.eval()
        with torch.no_grad():
            outputs = model(normalized_tensor)
            
            logits = outputs[0][0]
            preds = processor.post_process_logits(logits, processed_img["pixel_mask"].unsqueeze(0))
        
        # Filter predictions with a score higher than the confidence threshold
        filtered_preds = [pred for pred in preds if pred['score'] >= score_threshold]
    except Exception as e:
        raise RuntimeError(f"Failed to perform inference due to {e}.")
        

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