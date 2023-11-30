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
        if image_url.startswith("http"):
            image = Image.open(requests.get(image_url, stream=True).raw)
        else:
            image = Image.open(image_url)
    except OSError as e:
        print(e)
        return "Unable to open the image."
    
    # Load CLIP processor and model.
    processor = ChineseCLIPProcessor.from_pretrained("RenShaw/ChineseCLIP-image")
    model = ChineseCLIPModel.from_pretrained("RenShaw/ChineseCLIP-image").to("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Preprocess image.
    image = processor.vision_feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**image)
        prediction_scores = outputs.logits_per_image # (1, 20)
    
    predicted_tokens = processor.get_token_ids(category_labels).cuda() if torch.cuda.is_available() else processor.get_token_ids(category_labels)
    predicted_scores = torch.max(prediction_scores, dim=-1).values # (1, 20)
    preds = []
    
    for index in range(len(predicted_tokens)):
        preds.append((predicted_tokens[index], predicted_scores[0][index].item()))
    
    return max(preds, key=lambda item: item[1])[0]

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