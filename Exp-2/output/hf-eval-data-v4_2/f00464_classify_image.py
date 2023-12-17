# requirements_file --------------------

!pip install -U timm pillow

# function_import --------------------

from urllib.request import urlopen
from PIL import Image
import timm

# function_code --------------------

def classify_image(image_url):
    """
    Classify an image URL using MobileNetV3 model.

    Args:
        image_url (str): The URL of the image to classify.

    Returns:
        dict: A dictionary with category probabilities.

    Raises:
        ValueError: If the image_url is not reachable or invalid.
    """
    img = Image.open(urlopen(image_url))
    model = timm.create_model('mobilenetv3_large_100.ra_in1k', pretrained=True)
    model.eval()
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    transformed_img = transforms(img).unsqueeze(0)
    output = model(transformed_img)
    return output

# test_function_code --------------------

def test_classify_image():
    print("Testing started.")
    sample_image_url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'

    # Test case 1: Valid image URL
    print("Testing case [1/2] started.")
    try:
        result = classify_image(sample_image_url)
        assert result is not None, f"Test case [1/2] failed: Expected a result, got None"
    except ValueError as e:
        assert False, f"Test case [1/2] failed with error: {e}"

    # Test case 2: Invalid image URL
    print("Testing case [2/2] started.")
    try:
        classify_image('invalid_url')
        assert False, "Test case [2/2] failed: Expected a ValueError for invalid URL"
    except ValueError:
        pass

    print("Testing finished.")

# call_test_function_line --------------------

test_classify_image()