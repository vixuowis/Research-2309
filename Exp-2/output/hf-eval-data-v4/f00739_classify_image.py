# requirements_file --------------------

!pip install -U transformers==4.0.0 torch==1.9.0 Pillow==8.3.2 requests==2.26.0

# function_import --------------------

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

# function_code --------------------

def classify_image(image_url):
    """
    Download an image from the provided URL and classify the object in the image using the Vision Transformer (ViT) model.

    Args:
    - image_url (str): The URL of the image to classify.

    Returns:
    - str: The name of the predicted class.
    """
    try:
        # Download image from the internet
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        return f"Error downloading the image: {e}"

    # Initialize the image processor and the model
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    # Pre-process the image and prepare input tensor
    inputs = processor(images=image, return_tensors='pt')

    # Perform image classification
    outputs = model(**inputs)
    logits = outputs.logits

    # Get the predicted class index
    predicted_class_idx = logits.argmax(-1).item()

    # Return the predicted class
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_image():
    print("Testing started.")

    # Test case 1: Classify a known image of a labrador
    print("Testing case [1/3] started.")
    labrador_url = 'https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg'
    labrador_class = classify_image(labrador_url)
    assert labrador_class == "Labrador retriever", f"Test case [1/3] failed: Expected 'Labrador retriever', got {labrador_class}"
    print("Test case [1/3] passed.")

    # Test case 2: Classify a known image of a pizza
    print("Testing case [2/3] started.")
    pizza_url = 'https://upload.wikimedia.org/wikipedia/commons/a/a3/Eq_it-na_pizza-margherita_sep2005_sml.jpg'
    pizza_class = classify_image(pizza_url)
    assert pizza_class == "Pizza", f"Test case [2/3] failed: Expected 'Pizza', got {pizza_class}"
    print("Test case [2/3] passed.")

    # Test case 3: Use an invalid image URL
    print("Testing case [3/3] started.")
    invalid_url = 'http://invalid-url'
    result = classify_image(invalid_url)
    assert "Error downloading" in result, f"Test case [3/3] failed: Expected an error message, got {result}"
    print("Test case [3/3] passed.")

    print("Testing finished.")

# Run the test function
test_classify_image()