# function_import --------------------

import torch
from transformers import AutoModel
from PIL import Image

# function_code --------------------

def estimate_depth(image_path: str):
    """
    Estimate the depth information of a room from an image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        torch.Tensor: The depth information of each pixel in the image.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    image = Image.open(image_path)
    model = AutoModel.from_pretrained('sayakpaul/glpn-nyu-finetuned-diode-221116-104421')
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    return outputs

# test_function_code --------------------

def test_estimate_depth():
    """
    Test the estimate_depth function.
    """
    try:
        # Test with a valid image file
        outputs = estimate_depth('room_image.jpg')
        assert isinstance(outputs, torch.Tensor), 'The output should be a torch.Tensor'

        # Test with a non-existent image file
        try:
            outputs = estimate_depth('non_existent_image.jpg')
        except FileNotFoundError:
            pass
        else:
            assert False, 'The function should raise a FileNotFoundError for non-existent image files'
    except Exception as e:
        print(f'Test failed with exception: {e}')
    else:
        print('All tests passed')

# call_test_function_code --------------------

test_estimate_depth()