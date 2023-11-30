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
    
    # Setup the processor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = OwlViTProcessor.from_pretrained(model_name)
    model = OwlViTForObjectDetection.from_pretrained(model_name).to(device)
    
    # Get the image and preprocess it using PIL
    response = requests.get(url, stream=True).raw
    im = Image.open(response)
    img_pil = Image.new("RGB", (2048, 1728), color="black") # This is to handle images that do not have a multiple of 64 width and height
    img_pil.paste(im)
    
    inputs = processor(images=img_pil, return_tensors='pt')
    pixel_values = inputs['pixel_values'].to(device)
    
    # Perform object detection
    outputs = model(pixel_values)

    # Process the detections
    probs = outputs.logits_per_image[0][0].softmax(dim=-1).cpu().detach().numpy()
    bboxes = outputs.bboxes_per_image[0].cpu().detach().numpy()
    
    # Print the detections with their confidence scores and bounding box locations
    for text, prob in zip(texts, probs):
        filtered_probs = [prob[i] for i, label in enumerate(processor.get_labels()) if '{}'.format(text) == label]
        if len(filtered_probs) > 0:
            index = list(filtered_probs).index(max(list(filtered_probs)))
            score = round((max(filtered_probs)), 3)
            print('{}: {}'.format(text, score))
            
            if score >= score_threshold:
                xmin = int(round(bboxes[index][0]))
                ymin = int(round(bboxes[index][1]))
               

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