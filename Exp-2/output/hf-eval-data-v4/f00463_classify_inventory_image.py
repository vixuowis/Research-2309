# requirements_file --------------------

!pip install -U transformers torch pillow

# function_import --------------------

from transformers import AutoFeatureExtractor, RegNetForImageClassification
import torch
from PIL import Image

# function_code --------------------

def classify_inventory_image(image_path):
    """
    Classify the type of inventory item by its image using RegNet model.

    Args:
    image_path (str): The file path to the inventory image.

    Returns:
    str: The predicted label of the inventory item.
    """
    image = Image.open(image_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained('zuppif/regnet-y-040')
    model = RegNetForImageClassification.from_pretrained('zuppif/regnet-y-040')
    inputs = feature_extractor(image, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]

# test_function_code --------------------

def test_classify_inventory_image():
    print("Testing started.")
    # Assuming there is a test dataset available with labeled images
    test_image_path = 'test_inventory_image.jpg'
    expected_label = 'expected_label'

    print("Testing classification.")
    predicted_label = classify_inventory_image(test_image_path)
    assert predicted_label == expected_label, f"Test failed: Expected {{expected_label}}, got {{predicted_label}}"
    print("Testing finished.")