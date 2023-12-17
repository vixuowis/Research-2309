# requirements_file --------------------

!pip install -U transformers pillow requests

# function_import --------------------

from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests

# function_code --------------------

def classify_nature_image(image_url):
    # Load an image from the given URL
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Load preprocessor and model
    preprocessor = AutoImageProcessor.from_pretrained('google/mobilenet_v1_0.75_192')
    model = AutoModelForImageClassification.from_pretrained('google/mobilenet_v1_0.75_192')

    # Preprocess the image
    inputs = preprocessor(images=image, return_tensors='pt')

    # Classify the image
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    # Retrieve the predicted class label
    predicted_class = model.config.id2label[predicted_class_idx]

    return predicted_class

# test_function_code --------------------

def test_classify_nature_image():
    print('Testing classify_nature_image function')

    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    expected_class = 'elephant'  # Assuming we know the expected outcome

    # Run the classification
    predicted_class = classify_nature_image(image_url)

    # Check if the predicted class is correct
    assert predicted_class == expected_class, f'Test failed: Expected {expected_class}, got {predicted_class}'

    print('Test passed: Predicted class is {predicted_class}')

# Run the test
test_classify_nature_image()