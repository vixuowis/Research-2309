# requirements_file --------------------

!pip install -U torch transformers PIL

# function_import --------------------

from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# function_code --------------------

def detect_objects_in_kitchen(image_path, score_threshold=0.1):
    """
    Detects different objects in the kitchen using a pre-trained model.

    Args:
        image_path (str): Path to the image file that should be processed.
        score_threshold (float): Minimum score for the detections to be considered valid.

    Returns:
        list of tuples: A list of tuples containing detected objects and their details
        (label, score, bounding box).

    Raises:
        FileNotFoundError: If the image_path does not exist.
    """
    processor = OwlViTProcessor.from_pretrained('google/owlvit-large-patch14')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-large-patch14')
    image = Image.open(image_path)
    texts = [["a photo of a fruit", "a photo of a dish"]]
    inputs = processor(text=texts, images=image, return_tensors='pt')
    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
    detected_objects = []
    for i in range(len(texts)):
        for box, score, label in zip(results[i]["boxes"], results[i]["scores"], results[i]["labels"]):
            if score >= score_threshold:
                box = [round(point, 2) for point in box.tolist()]
                detected_objects.append((texts[0][label], round(score.item(), 3), box))
    return detected_objects

# test_function_code --------------------

def test_detect_objects_in_kitchen():
    print("Testing started.")
    # Assuming we have a function load_dataset() that loads our kitchen image dataset
    dataset = load_dataset("kitchen_data")
    sample_image_path = dataset[0]  # For the sake of example

    # Test case 1: Check if the function returns a list
    print("Testing case [1/3] started.")
    result = detect_objects_in_kitchen(sample_image_path)
    assert isinstance(result, list), f"Test case [1/3] failed: Expected result type list, got {type(result)}"

    # Test case 2: Check if the function raises FileNotFoundError on wrong path
    print("Testing case [2/3] started.")
    try:
        detect_objects_in_kitchen("nonexistent_image.jpg")
        assert False, "Test case [2/3] failed: FileNotFoundError not raised on non-existent image path"
    except FileNotFoundError:
        pass  # Expected exception

    # Test case 3: Check if the result list elements are tuples
    print("Testing case [3/3] started.")
    if result:
        assert all(isinstance(item, tuple) for item in result), f"Test case [3/3] failed: Expected elements of result to be tuples"
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_objects_in_kitchen()