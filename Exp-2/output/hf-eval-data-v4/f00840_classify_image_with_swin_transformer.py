# requirements_file --------------------

!pip install -U transformers PIL requests

# function_import --------------------

from transformers import AutoFeatureExtractor, SwinForImageClassification
from PIL import Image
import requests

# function_code --------------------

def classify_image_with_swin_transformer(image_url: str) -> str:
    """
    Classify an image using the Swin Transformer model.

    Parameters:
    image_url (str): URL of the image to classify.

    Returns:
    str: The predicted class label of the image.
    """
    # Load the image from the web
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Initialize the feature extractor and model
    feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
    model = SwinForImageClassification.from_pretrained('microsoft/swin-tiny-patch4-window7-224')

    # Preprocess the image and prepare it for the model
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Get the model's prediction
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    # Retrieve the predicted class label
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_image_with_swin_transformer():
    print("Testing started.")

    # URL of a sample image
    sample_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    # Expected result for the sample image
    expected_class_label = 'elephant'

    # Test case: Classify the sample image
    print("Testing classification of sample image.")
    predicted_label = classify_image_with_swin_transformer(sample_image_url)
    assert predicted_label == expected_class_label, f"Test failed: expected {expected_class_label}, got {predicted_label}"
    print("Testing finished.")

    # Run the test
    test_classify_image_with_swin_transformer()