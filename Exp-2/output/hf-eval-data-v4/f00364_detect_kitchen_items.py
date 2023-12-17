# requirements_file --------------------

!pip install -U torch transformers Pillow

# function_import --------------------

from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# function_code --------------------

def detect_kitchen_items(image_path, queries, score_threshold=0.1):
    # Load the processor and model
    processor = OwlViTProcessor.from_pretrained('google/owlvit-large-patch14')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-large-patch14')
    
    # Open the image
    image = Image.open(image_path)
    
    # Prepare the inputs for the model
    inputs = processor(text=queries, images=image, return_tensors='pt')
    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
    
    # Filter results by the score threshold
    detections = []
    for i in range(len(queries)):
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        for box, score, label in zip(boxes, scores, labels):
            if score >= score_threshold:
                detection = {
                    'query': queries[i],
                    'box': [round(b, 2) for b in box.tolist()],
                    'score': round(score.item(), 3)
                }
                detections.append(detection)
    return detections

# test_function_code --------------------

def test_detect_kitchen_items():
    print("Testing started.")
    # Assume `kitchen_image.jpg` is in the test dataset
    test_image_path = 'kitchen_image.jpg'
    test_queries = [["a photo of a fruit", "a photo of a dish"]]

    # Test case 1: Check if the function works without errors
    print("Testing case [1/1] started.")
    detections = detect_kitchen_items(test_image_path, test_queries)
    assert isinstance(detections, list), f"Test case [1/1] failed: Expected a list, got {type(detections)}"
    print("Detected items:", detections)
    print("Testing finished.")

# Run the test function
test_detect_kitchen_items()