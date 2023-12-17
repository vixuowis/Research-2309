# requirements_file --------------------

!pip install -U transformers torch pillow requests

# function_import --------------------

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

# function_code --------------------

def classify_image(img_url: str) -> str:
    """
    Classify an image given its URL using a pre-trained Vision Transformer (ViT).
    
    Args:
        img_url (str): The URL of the image to be classified.
    
    Returns:
        str: The predicted class label.
    
    Raises:
        ValueError: If the image cannot be processed or the URL is invalid.
    """
    try:
        # Download the image
        image = Image.open(requests.get(img_url, stream=True).raw)

        # Initialize the image processor
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

        # Load the pre-trained model
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

        # Prepare the image
        inputs = processor(images=image, return_tensors='pt')

        # Perform image classification
        outputs = model(**inputs)
        logits = outputs.logits

        # Extract the predicted class ID
        predicted_class_idx = logits.argmax(-1).item()

        # Convert the ID to the class label
        predicted_class = model.config.id2label[predicted_class_idx]
        return predicted_class
    except Exception as e:
        raise ValueError(f"An error occurred during image classification: {e}")

# test_function_code --------------------

def test_classify_image():
    print("Testing started.")
    # Testing scenario with a known image URL
    test_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    # Expected label for the provided image
    expected_label = 'ENTER_EXPECTED_LABEL_HERE'

    # Test case 1: Valid Image URL
    print("Testing case [1/1] started.")
    predicted_label = classify_image(test_url)
    assert predicted_label == expected_label, f"Test case [1/1] failed: Expected '{{expected_label}}', got '{{predicted_label}}'"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_image()