# function_import --------------------

from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# function_code --------------------

def detect_kitchen_objects(image_path: str, score_threshold: float = 0.1):
    """
    Detects different objects in a kitchen using a pre-trained model.

    Args:
        image_path (str): The path to the image file.
        score_threshold (float, optional): The confidence score threshold for object detection. Defaults to 0.1.

    Returns:
        None. Prints the detected objects, their confidence scores, and their locations.
    """
    processor = OwlViTProcessor.from_pretrained('google/owlvit-large-patch14')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-large-patch14')
    image = Image.open(image_path)
    texts = [["a photo of a fruit", "a photo of a dish"]]
    inputs = processor(text=texts, images=image, return_tensors='pt')
    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    for i in range(len(texts)):
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            if score >= score_threshold:
                print(f"Detected {texts[0][label]} with confidence {round(score.item(), 3)} at location {box}")

# test_function_code --------------------

def test_detect_kitchen_objects():
    """
    Tests the detect_kitchen_objects function.
    """
    detect_kitchen_objects('test_image.jpg', 0.1)

# call_test_function_code --------------------

test_detect_kitchen_objects()