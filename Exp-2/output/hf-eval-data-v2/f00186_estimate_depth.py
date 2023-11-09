# function_import --------------------

from transformers import AutoModel
from PIL import Image

# function_code --------------------

def estimate_depth(image_path):
    """
    This function estimates the depth information of a room using a pre-trained model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        Tensor: The estimated depth information.
    """
    # Load the image
    image = Image.open(image_path)

    # Load the pre-trained model
    model = AutoModel.from_pretrained('sayakpaul/glpn-nyu-finetuned-diode-221116-104421')

    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Estimate the depth
    outputs = model(**inputs)

    return outputs

# test_function_code --------------------

def test_estimate_depth():
    """
    This function tests the estimate_depth function.
    """
    # Define the image path
    image_path = 'test_image.jpg'

    # Call the function
    outputs = estimate_depth(image_path)

    # Assert the outputs
    assert outputs is not None
    assert isinstance(outputs, torch.Tensor)

# call_test_function_code --------------------

test_estimate_depth()