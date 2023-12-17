# requirements_file --------------------

!pip install -U PIL requests transformers

# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def classify_image_as_cat_or_dog(image_url):
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        return str(e)

    inputs = processor(text=['a photo of a cat', 'a photo of a dog'], images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    _, predicted_class_idx = probs.max(dim=1)
    labels = ['cat', 'dog']
    return labels[predicted_class_idx.item()]

# test_function_code --------------------

def test_classify_image_as_cat_or_dog():

    test_cases = [
        ('http://example.com/cat.jpg', 'cat'),
        ('http://example.com/dog.jpg', 'dog'),
        ('http://example.com/invalid.jpg', 'error')
    ]

    for i, (url, expected) in enumerate(test_cases):
        result = classify_image_as_cat_or_dog(url)
        assert result == expected, f"Test case [{i+1}] failed: Expected {expected}, got {result}"

    print('All test cases passed!')

test_classify_image_as_cat_or_dog()