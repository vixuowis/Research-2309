# function_import --------------------

from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
import torch
from datasets import load_dataset

# function_code --------------------

def classify_product(image):
    """
    Classify the type of product in an image using a pretrained ConvNext model.

    Args:
        image (str): The path to the image file.

    Returns:
        str: The predicted label of the product.
    """
    # Load the feature extractor and model
    feature_extractor = ConvNextFeatureExtractor.from_pretrained('facebook/convnext-large-224')
    model = ConvNextForImageClassification.from_pretrained('facebook/convnext-large-224')

    # Preprocess the image
    inputs = feature_extractor(image, return_tensors='pt')

    # Generate logits representing the probability of each object category
    with torch.no_grad():
        logits = model(**inputs).logits

    # Identify the most likely object class
    predicted_label = logits.argmax(-1).item()

    return model.config.id2label[predicted_label]

# test_function_code --------------------

def test_classify_product():
    """
    Test the classify_product function with a sample image.
    """
    # Load a sample image from the dataset
    dataset = load_dataset('your_dataset')
    image = dataset['test']['image'][0]

    # Call the classify_product function
    predicted_label = classify_product(image)

    # Assert that the function returns a string (the predicted label)
    assert isinstance(predicted_label, str)

# call_test_function_code --------------------

test_classify_product()