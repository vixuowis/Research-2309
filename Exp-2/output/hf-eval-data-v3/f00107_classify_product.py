# function_import --------------------

from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
import torch
from datasets import load_dataset

# function_code --------------------

def classify_product(dataset_name: str, image_index: int = 0):
    """
    Classify the product image using pretrained ConvNext model.

    Args:
        dataset_name (str): The name of the dataset to be loaded.
        image_index (int, optional): The index of the image in the dataset to be classified. Defaults to 0.

    Returns:
        str: The predicted label of the product image.
    """
    dataset = load_dataset(dataset_name)
    image = dataset['test']['image'][image_index]
    feature_extractor = ConvNextFeatureExtractor.from_pretrained('facebook/convnext-large-224')
    model = ConvNextForImageClassification.from_pretrained('facebook/convnext-large-224')
    inputs = feature_extractor(image, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]

# test_function_code --------------------

def test_classify_product():
    """
    Test the classify_product function.
    """
    assert classify_product('huggingface/cats-image', 0) == 'cat'
    assert classify_product('huggingface/dogs-image', 0) == 'dog'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_product()