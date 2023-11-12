# function_import --------------------

from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# function_code --------------------

def classify_product_image(image_url: str, category_labels: list) -> str:
    """
    Classify a product image into one of the given categories using a pre-trained ChineseCLIPModel.

    Args:
        image_url (str): The URL or file path of the product image to be classified.
        category_labels (list): A list of category labels for classification.

    Returns:
        str: The predicted category for the product image.

    Raises:
        OSError: If there is a problem with the file path or the image cannot be opened.
    """
    model = ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14')
    processor = ChineseCLIPProcessor.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14')

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
    category = category_labels[category_index]

    return category

# test_function_code --------------------

def test_classify_product_image():
    """Test the classify_product_image function."""
    image_url = 'https://placekitten.com/200/300'
    category_labels = ['cat', 'dog', 'bird']
    predicted_category = classify_product_image(image_url, category_labels)
    assert predicted_category in category_labels, 'The predicted category is not in the category labels.'

    image_url = 'https://placekitten.com/200/301'
    category_labels = ['cat', 'dog', 'bird']
    predicted_category = classify_product_image(image_url, category_labels)
    assert predicted_category in category_labels, 'The predicted category is not in the category labels.'

    image_url = 'https://placekitten.com/200/302'
    category_labels = ['cat', 'dog', 'bird']
    predicted_category = classify_product_image(image_url, category_labels)
    assert predicted_category in category_labels, 'The predicted category is not in the category labels.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_product_image()