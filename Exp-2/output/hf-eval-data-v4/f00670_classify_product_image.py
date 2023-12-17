# requirements_file --------------------

!pip install -U transformers PIL requests

# function_import --------------------

from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# function_code --------------------

def classify_product_image(image_url, category_labels):
    # Load the pre-trained ChineseCLIPModel
    model = ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14')
    processor = ChineseCLIPProcessor.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14')

    # Load the image
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Process the image features
    inputs = processor(images=image, return_tensors='pt')
    image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    # Process the text features
    inputs = processor(text=category_labels, padding=True, return_tensors='pt')
    text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    # Compute the similarity score between the image and text labels
    inputs = processor(text=category_labels, images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    category_index = probs.argmax().item()

    # Return the predicted category
    return category_labels[category_index]

# test_function_code --------------------

def test_classify_product_image():
    print("Testing classify_product_image function.")

    # Test with a sample image URL and category labels
    image_url = 'https://example.com/sample_image.jpg'  # Replace with a real image URL
    category_labels = ['Electronics', 'Apparel', 'Toys', 'Food']

    predicted_category = classify_product_image(image_url, category_labels)
    assert predicted_category in category_labels, f"The predicted category '{predicted_category}' is not among the known category labels."

    print("All tests passed!")