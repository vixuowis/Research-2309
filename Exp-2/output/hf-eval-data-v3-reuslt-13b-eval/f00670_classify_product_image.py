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
        # 1. Download and load the model and tokenizer files.
        processor = ChineseCLIPProcessor.from_pretrained("shibing624/text-clip-chinese-roberta-base-distil")
        model = ChineseCLIPModel.from_pretrained("shibing624/text-clip-chinese-roberta-base-distil", processor=processor)

        # 2. Download and load the image to be classified.
        response = requests.get(image_url, stream=True)
        with open('product_image.jpg', 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        image = Image.open("product_image.jpg")
        inputs = processor(text=category_labels, images=image, return_tensors="pt", padding=True)

        # 3. Make the classification prediction.
        outputs = model(**inputs)
        logits_per_text = outputs.logits_per_text
        predicted_label = category_labels[logits_per_text[0].argmax()]
    except OSError as err:
        print("OSError: {0}".format(err))
        raise
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