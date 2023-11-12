# function_import --------------------

from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

# function_code --------------------

def is_good_time_to_visit(image_path):
    """
    Determine whether it is a good time to visit a Chinese historical site based on an image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        bool: True if it is a good time to visit, False otherwise.
    """
    model = ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-base-patch16')
    processor = ChineseCLIPProcessor.from_pretrained('OFA-Sys/chinese-clip-vit-base-patch16')

    image = Image.open(image_path)
    texts = ["好的参观时间", "不是好的参观时间"]

    inputs = processor(images=image, text=texts, return_tensors='pt')
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).tolist()
    result = dict(zip(texts, probs[0]))

    return result['好的参观时间'] > result['不是好的参观时间']

# test_function_code --------------------

def test_is_good_time_to_visit():
    """
    Test the function is_good_time_to_visit.
    """
    assert is_good_time_to_visit('test_images/good_time.jpg') == True
    assert is_good_time_to_visit('test_images/bad_time.jpg') == False
    assert is_good_time_to_visit('test_images/uncertain_time.jpg') is not None

    return 'All Tests Passed'

# call_test_function_code --------------------

test_is_good_time_to_visit()