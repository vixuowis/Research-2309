# requirements_file --------------------

!pip install -U pillow requests transformers

# function_import --------------------

from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

model = ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14')
processor = ChineseCLIPProcessor.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14')

# function_code --------------------

def classify_product_image(image_url, category_labels):
    """
    Classifies a product image into one of the given category labels using zero-shot classification.

    Args:
        image_url (str): URL or filepath containing the product image.
        category_labels (list): List of category labels to classify the image into.

    Returns:
        str: The predicted category label for the image.

    Raises:
        ValueError: If `category_labels` is empty.
    """
    if not category_labels:
        raise ValueError('Category labels list cannot be empty.')

    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(images=image, return_tensors='pt')
    image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    inputs = processor(text=category_labels, padding=True, return_tensors='pt')
    text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    inputs = processor(text=category_labels, images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    category_index = probs.argmax().item()
    return category_labels[category_index]

# test_function_code --------------------

def test_classify_product_image():
    print("Testing started.")
    # Assume a placeholder image URL and category labels for testing
    image_url = "https://example.com/sample_image.jpg"
    category_labels = ['Electronics', 'Clothing', 'Toys']

    # Testing case 1: Valid inputs
    print("Testing case [1/3] started.")
    predicted_category = classify_product_image(image_url, category_labels)
    assert predicted_category in category_labels, f"Test case [1/3] failed: Predicted category not in the expected list."

    # Testing case 2: Empty category labels
    print("Testing case [2/3] started.")
    try:
        classify_product_image(image_url, [])
        assert False, "Test case [2/3] failed: ValueError not raised for empty category labels."
    except ValueError:
        pass

    # Testing case 3: Invalid image URL
    print("Testing case [3/3] started.")
    # Here it's assumed that the test environment cannot fetch the image and raises an IOError
    try:
        classify_product_image("invalid_url", category_labels)
        assert False, "Test case [3/3] failed: IOError not raised for invalid image URL."
    except IOError:
        pass

    print("Testing finished.")

# call_test_function_line --------------------

test_classify_product_image()