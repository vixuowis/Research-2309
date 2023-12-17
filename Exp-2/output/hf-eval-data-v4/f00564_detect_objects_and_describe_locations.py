# requirements_file --------------------

!pip install -U torch transformers PIL requests

# function_import --------------------

import requests
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# function_code --------------------

def detect_objects_and_describe_locations(image_url, texts, score_threshold=0.1):
    """
    Detect objects in an image provided by a user and describe their locations.

    Parameters:
        image_url (str): URL of the image to process.
        texts (list of str): List of text queries describing expected objects in the image.
        score_threshold (float): Threshold for object detection scores.

    Returns:
        list of tuple: Detected object data with labels, scores, and bounding boxes.
    """
    # Load the processor and model
    processor = OwlViTProcessor.from_pretrained('google/owlvit-large-patch14')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-large-patch14')

    # Load the image from URL
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Prepare the input tensors
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Post-process the predictions
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    detected_objects = []
    for i, text in enumerate(texts):
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        for box, score, label in zip(boxes, scores, labels):
            if score >= score_threshold:
                detected_objects.append((text, round(score.item(), 3), [round(i, 2) for i in box.tolist()]))

    return detected_objects

# test_function_code --------------------

def test_detect_objects_and_describe_locations():
    print("Testing detect_objects_and_describe_locations function.")

    # Assuming we have a function that loads test image URLs and expected results
    test_images = load_test_images()

    for img_url, expected in test_images:
        print(f"Testing with image: {img_url}")
        result = detect_objects_and_describe_locations(img_url, texts=['a photo of a cat', 'a photo of a dog'])
        assert len(result) == expected['object_count'], f"Incorrect number of objects detected for image {img_url}."
        for obj_data, exp_data in zip(result, expected['data']):
            assert obj_data == exp_data, f"Mismatch in detected object data for image {img_url}: {obj_data} != {exp_data}"
        print(f"Test for image {img_url} passed!")