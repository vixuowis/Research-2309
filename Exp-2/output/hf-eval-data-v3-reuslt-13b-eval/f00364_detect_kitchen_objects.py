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
    # Load the pretrained model from Hugging Face.
    try:
        processor = OwlViTProcessor.from_pretrained("dandelin/owlvit-object-classification")
        model = OwlViTForObjectDetection.from_pretrained("dandelin/owlvit-object-classification")
    except:
        raise RuntimeError("Could not load the pretrained model.")

    # Load and process the image file.
    if os.path.isfile(image_path):
        try:
            pil_img = Image.open(image_path).convert('RGB')
        except:
            raise RuntimeError("Could not load the image file.")
    else:
        raise FileNotFoundError("Image file does not exist")
    
    # Make the prediction and print results to console.
    pixel_values = processor(pil_img, return_tensors="pt").pixel_values
    outputs = model(pixel_values)
    logits = outputs.logits[0]["logits"]
    
    if torch.numel(logits) == 1:
        print("No objects detected.")
    else:
        print("Objects found:\n")
        
        for i in range(torch.numel(logits)):
            score = logits[i]
            
            if float(score) > score_threshold:
                class_name = processor.id2label[int(i / 80)]
                
                print("{}: {}".format(class_name, round((float(score)*100), 3)))

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