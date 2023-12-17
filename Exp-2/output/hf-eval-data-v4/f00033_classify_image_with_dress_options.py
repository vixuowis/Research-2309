# requirements_file --------------------

!pip install -U requests, torch, Pillow, transformers

# function_import --------------------

import requests
from PIL import Image
from transformers import AlignProcessor, AlignModel

# function_code --------------------

def classify_image_with_dress_options(image_url, candidate_labels):
    # Load the necessary models
    processor = AlignProcessor.from_pretrained('kakaobrain/align-base')
    model = AlignModel.from_pretrained('kakaobrain/align-base')

    # Load and process the image
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(text=candidate_labels, images=image, return_tensors='pt')

    # Perform zero-shot classification
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # Return the probabilities for each dress option
    return probs.tolist()

# test_function_code --------------------

def test_classify_image_with_dress_options():
    print("Testing started.")
    test_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    test_labels = ['an image of casual dressing', 'an image of formal dressing']

    # Test case: Dress classifications for a given image
    print("Testing case [1/1] started.")
    probabilities = classify_image_with_dress_options(test_url, test_labels)
    assert len(probabilities) == len(test_labels), f"Test case [1/1] failed: Expected {len(test_labels)} probabilities, got {len(probabilities)}"
    print(f"Probabilities: {probabilities}")
    print("Testing finished.")

# Run the test function
test_classify_image_with_dress_options()