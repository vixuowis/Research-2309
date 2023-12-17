# requirements_file --------------------

!pip install -U diffusers pillow

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_vintage_image(model_name: str) -> 'Image':
    """
    Generate a high-quality vintage image using a pre-trained model.

    Args:
        model_name (str): The name of the pre-trained model to use.

    Returns:
        Image: A PIL Image object of the generated vintage image.

    Raises:
        ValueError: If the model name is empty.
    """
    if not model_name:
        raise ValueError('Model name must not be empty.')
    pipeline = DDPMPipeline.from_pretrained(model_name)
    return pipeline().images[0]

# test_function_code --------------------

from PIL import Image
from io import BytesIO

def test_generate_vintage_image():
    print("Testing started.")

    model_name = 'pravsels/ddpm-ffhq-vintage-finetuned-vintage-3epochs'
    print("Testing case [1/1] started.")
    image = generate_vintage_image(model_name)
    assert isinstance(image, Image.Image), f"Test case [1/1] failed: Expected PIL Image, got {type(image)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_vintage_image()