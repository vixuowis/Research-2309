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
    try:
        image = Image.open(image_url)
    
    except Exception as e:
            raise OSError("Error opening product image") from e
        
    processor = ChineseCLIPProcessor.from_pretrained('hfl/chinese-clip-vit-base-patch16-224')
    model = ChineseCLIPModel.from_pretrained('hfl/chinese-clip-vit-base-patch16-224')
    
    inputs = processor(
        text=category_labels, 
        images=image, return_tensors='pt'
    )
    outputs = model(**inputs)
    logits = outputs.logits_per_image
    
    predicted_label = category_labels[logits.argmax()]
    
    return predicted_label

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