# requirements_file --------------------

import subprocess

requirements = ["transformers", "pillow", "requests"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests

# function_code --------------------

def classify_houseplant(image_url):
    """
    Classifies the type of houseplant in the image located at image_url.

    Args:
        image_url (str): The URL of the image to be classified.

    Returns:
        str: The predicted type of the houseplant.

    Raises:
        ValueError: If image_url is not valid or image cannot be opened.
    """

    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise ValueError(f'Unable to open image. Reason: {str(e)}')

    preprocessor = AutoImageProcessor.from_pretrained('google/mobilenet_v1_0.75_192')
    model = AutoModelForImageClassification.from_pretrained('google/mobilenet_v1_0.75_192')
    inputs = preprocessor(images=image, return_tensors='pt')

    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_houseplant():
    print("Testing started.")
    # Assuming there is a function named `load_dataset` available
    sample_data = load_dataset("houseplants")[0]  # 从数据集中抽取一个样本

    # 测试用例 1
    print("Testing case [1/1] started.")
    image_url = sample_data['image']
    predicted_type = classify_houseplant(image_url)
    assert predicted_type in ['cactus', 'fern', 'succulent'], f"Test case [1/1] failed: Unexpected plant type {predicted_type}."
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_houseplant()