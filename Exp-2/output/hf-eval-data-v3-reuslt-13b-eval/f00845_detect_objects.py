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
    
    model = OwlViTForObjectDetection.from_pretrained(model_name)
    processor = OwlViTProcessor.from_pretrained(model_name)
    
    # Load image and prepare it for detection
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    inputs = processor(texts=texts, images=[img], return_tensors="pt", padding='max_length', max_length=128, return_processor=False)
    
    # Predict the objects in the image
    outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.sigmoid(logits).detach().cpu()
    
    # Filter out weak detections, and extract their confidence scores and bounding box locations
    conf_score = list()
    x0_location = list()
    y0_location = list()
    x1_location = list()
    y1_location = list()
    
    for i in range(len(texts)):
        if preds[i][0] > score_threshold: # only include detections with a confidence above the threshold
            conf_score.append(preds[i][0])
            
            bbox = outputs.pred_boxes[0].detach().cpu() # the bounding boxes are tensors of shape (1,60,4) 
            x0_location.append(bbox[0][i][0])
            y0_location.append(bbox[0][i][1])
            x1_location.append(bbox[0][i][2])
            y1_location.append(bbox[0][i][3])
    
    print('Detected objects:')
    for i in range(len(conf_score)):
        print(texts[i], conf_score[i], '(', x0_location[i], y0_location[i], ',', x1_location[i], y1_location[i], ')')

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