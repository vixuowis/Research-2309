# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch", "datasets"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
import torch
from datasets import load_dataset

# function_code --------------------

def classify_product_images(dataset_path):
    '''
    Classify the product images using a pretrained ConvNeXT model.

    Args:
        dataset_path (str): The path to the dataset containing product images.

    Returns:
        list: The list of predicted labels for the images.

    Raises:
        ValueError: If the dataset is not found or empty.
        RuntimeError: If the model or feature extractor loading fails.
    '''
    dataset = load_dataset(dataset_path)
    if not dataset or 'test' not in dataset or len(dataset['test']) == 0:
        raise ValueError('The dataset is not found or empty.')

    feature_extractor = ConvNextFeatureExtractor.from_pretrained('facebook/convnext-large-224')
    model = ConvNextForImageClassification.from_pretrained('facebook/convnext-large-224')

    predictions = []
    for image in dataset['test']['image']:
        inputs = feature_extractor(image, return_tensors='pt')
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_label = logits.argmax(-1).item()
            predictions.append(model.config.id2label[predicted_label])

    return predictions

# test_function_code --------------------

def test_classify_product_images():
    print("Testing started.")
    dataset = load_dataset('huggingface/cats-image')

    # Test case 1: Dataset has images
    print("Testing case [1/1] started.")
    predictions = classify_product_images('huggingface/cats-image')
    assert predictions, f"Test case [1/1] failed: Expected predictions, got {predictions}"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_product_images()