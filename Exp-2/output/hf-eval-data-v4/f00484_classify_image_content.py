# requirements_file --------------------

!pip install -U PIL, requests, transformers

# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def classify_image_content(image_url: str):
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')

    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(text=['a photo of a cat', 'a photo of a dog'], images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    categories = ['cat', 'dog']
    classification_results = dict(zip(categories, probs.tolist()[0]))
    return classification_results

# test_function_code --------------------

def test_classify_image_content():
    print('Testing the classify_image_content function...')

    # Test case 1: Image with a cat
    print('Testing case 1: Image with a cat')
    cat_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    result = classify_image_content(cat_image_url)
    print('Classification result:', result)
    assert max(result, key=result.get) == 'cat', 'Test case 1 failed: Expected category is cat.'

    # Test case 2: Image with a dog
    print('Testing case 2: Image with a dog')
    dog_image_url = 'http://images.cocodataset.org/val2017/000000575149.jpg'
    result = classify_image_content(dog_image_url)
    print('Classification result:', result)
    assert max(result, key=result.get) == 'dog', 'Test case 2 failed: Expected category is dog.'

    print('All tests passed.')
    return 'All tests passed.'