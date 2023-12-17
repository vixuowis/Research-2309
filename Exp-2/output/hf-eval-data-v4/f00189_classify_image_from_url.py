# requirements_file --------------------

!pip install -U timm

# function_import --------------------

from urllib.request import urlopen
from PIL import Image
import timm

# function_code --------------------

def classify_image_from_url(img_url):
    """
    Classify an image from a given URL into one of a thousand categories.

    :param img_url: str, The URL of the image to classify.
    :return: dict, The probabilities for each category.
    """
    # Load the image from the URL
    img = Image.open(urlopen(img_url))

    # Load the pretrained model
    model = timm.create_model('convnext_base.fb_in1k', pretrained=True)
    model.eval()

    # Set up the data transformation
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # Perform image transformation and classification
    transformed_img = transforms(img).unsqueeze(0)
    output = model(transformed_img)

    # Convert the output to probabilities
    probabilities = output.softmax(dim=1).squeeze().tolist()

    return probabilities

# test_function_code --------------------

def test_classify_image_from_url():
    print("Testing started.")
    # Test case-1: Check if the function returns a list
    print("Testing case [1/1] started.")
    test_url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    result = classify_image_from_url(test_url)
    assert isinstance(result, list), f"Test case [1/1] failed: Expected result type list, got {type(result)}"
    print("Testing case [1/1] passed.")
    print("Testing finished.")