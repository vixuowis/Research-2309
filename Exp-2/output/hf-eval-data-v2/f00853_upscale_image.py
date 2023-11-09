# function_import --------------------

from transformers import Swin2SRForConditionalGeneration
import torch

# function_code --------------------

def upscale_image(low_resolution_tensor):
    """
    This function upscales a low-resolution image to twice its size using a pretrained Swin2SR model.

    Args:
        low_resolution_tensor (torch.Tensor): The low-resolution image tensor.

    Returns:
        torch.Tensor: The upscaled image tensor.
    """
    # Load the pretrained Swin2SR model
    model = Swin2SRForConditionalGeneration.from_pretrained('conde/Swin2SR-lightweight-x2-64')

    # Pass the tensor through the model to obtain the upscaled image
    upscaled_tensor = model(low_resolution_tensor)

    return upscaled_tensor

# test_function_code --------------------

def test_upscale_image():
    """
    This function tests the `upscale_image` function by upscaling a low-resolution image and checking the size of the output.
    """
    # Create a low-resolution image tensor
    low_resolution_tensor = torch.rand(1, 3, 64, 64)

    # Upscale the image
    upscaled_tensor = upscale_image(low_resolution_tensor)

    # Check the size of the upscaled image
    assert upscaled_tensor.size() == torch.Size([1, 3, 128, 128]), 'The upscaled image size is incorrect.'

# call_test_function_code --------------------

test_upscale_image()