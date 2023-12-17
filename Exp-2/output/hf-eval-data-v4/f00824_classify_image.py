# requirements_file --------------------

!pip install -U transformers PIL requests

# function_import --------------------

from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests

# function_code --------------------

def classify_image(image_url):
    """
    Classify an image from a given URL using the pretrained Vision Transformer model.

    Parameters
    ----------
    image_url : str
        The URL of the image to classify.

    Returns
    -------
    dict
        The classification results including the last hidden states.
    """
    image = Image.open(requests.get(image_url, stream=True).raw)
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    # Normally, you would decode the last_hidden_states to actual labels. Here, we just return it
    return {'last_hidden_states': last_hidden_states}


# test_function_code --------------------

def test_classify_image():
    print("Testing classify_image function.")
    test_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    # Test case 1: Checking if function returns a non-empty result
    print("Testing case [1/1] started.")
    result = classify_image(test_image_url)
    assert result is not None and 'last_hidden_states' in result, f"Test case [1/1] failed: Function did not return the expected key 'last_hidden_states'."
    print("Testing case [1/1] passed.")
    print("Testing finished.")

# Run the test function
test_classify_image()