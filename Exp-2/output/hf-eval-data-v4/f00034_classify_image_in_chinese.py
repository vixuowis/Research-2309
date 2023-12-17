# requirements_file --------------------

!pip install -U PIL, requests, transformers

# function_import --------------------

from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# function_code --------------------

def classify_image_in_chinese(image_url, text_descriptions):
    """
    Classifies an image by comparing it with a list of text descriptions in Chinese.

    :param image_url: The URL of the image to classify.
    :param text_descriptions: A list of text descriptions in Chinese.
    :return: A dictionary containing the probabilities for each text description.
    """
    model = ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14-336px')
    processor = ChineseCLIPProcessor.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14-336px')
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(text=text_descriptions, images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    result = {desc: float(prob) for desc, prob in zip(text_descriptions, probs.squeeze())}
    return result

# test_function_code --------------------

def test_classification_example():
    print("Testing classify_image_in_chinese function.")
    image_url = 'https://example.com/image.jpg'
    text_descriptions = ['文本描述1', '文本描述2', '文本描述3']
    classification_result = classify_image_in_chinese(image_url, text_descriptions)
    assert isinstance(classification_result, dict), "The function should return a dictionary."
    assert len(classification_result) == len(text_descriptions), "The result dictionary should have the same number of keys as the text descriptions list."
    for desc in text_descriptions:
        assert desc in classification_result, f"The description {desc} should be a key in the result dictionary."
        assert isinstance(classification_result[desc], float), "Each value in the result should be a float representing the probability."
    print("Testing completed successfully.")

test_classification_example()