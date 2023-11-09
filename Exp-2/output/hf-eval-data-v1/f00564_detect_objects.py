import requests
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

def detect_objects(image_url):
    '''
    Detect objects in an image using the OwlViTForObjectDetection model from Hugging Face Transformers.
    Args:
    image_url (str): URL of the image to process.
    Returns:
    str: Detected objects and their locations.
    '''
    processor = OwlViTProcessor.from_pretrained('google/owlvit-large-patch14')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-large-patch14')
    image = Image.open(requests.get(image_url, stream=True).raw)
    texts = ['a photo of a cat', 'a photo of a dog']
    inputs = processor(text=texts, images=image, return_tensors='pt')
    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
    score_threshold = 0.1
    detected_objects = []
    for i, text in enumerate(texts):
        boxes, scores, labels = results[i]['boxes'], results[i]['scores'], results[i]['labels']
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            if score >= score_threshold:
                detected_objects.append(f'Detected {text} with confidence {round(score.item(), 3)} at location {box}')
    return '\n'.join(detected_objects)