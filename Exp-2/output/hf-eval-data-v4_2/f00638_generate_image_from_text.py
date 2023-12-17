# requirements_file --------------------

!pip install -U diffusers transformers accelerate scipy safetensors

# function_import --------------------

from diffusers import StableDiffusionInpaintPipeline

# function_code --------------------

def generate_image_from_text(prompt, image=None, mask_image=None):
    """
    Generate an image based on a text prompt using the StableDiffusionInpaintPipeline.

    Args:
        prompt (str): A text description to generate an image from.
        image (PIL.Image, optional): The base image to apply inpainting, if needed. Defaults to None.
        mask_image (PIL.Image, optional): The mask image for inpainting areas. Defaults to None.

    Returns:
        PIL.Image: Generated image based on the text prompt.

    Raises:
        ValueError: If the prompt is empty or None.
    """
    if not prompt:
        raise ValueError('The prompt cannot be empty.')
    pipe = StableDiffusionInpaintPipeline.from_pretrained('stabilityai/stable-diffusion-2-inpainting', torch_dtype=torch.float16)
    pipe.to('cuda')
    output_image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
    return output_image

# test_function_code --------------------

def test_generate_image_from_text():
    print("Testing started.")

    # Test case 1: Valid prompt with no images
    print("Testing case [1/3] started.")
    prompt = "A futuristic cityscape at night with neon lights."
    try:
        result_image = generate_image_from_text(prompt)
        assert result_image is not None, f"Test case [1/3] failed: Expected a generated image, but got None."
    except Exception as e:
        assert False, f"Test case [1/3] failed with an exception: {e}"

    # Test case 2: Empty prompt
    print("Testing case [2/3] started.")
    prompt = ""
    try:
        result_image = generate_image_from_text(prompt)
        assert False, "Test case [2/3] failed: Expected ValueError for empty prompt."
    except ValueError as e:
        assert str(e) == 'The prompt cannot be empty.', f"Test case [2/3] failed: {e}"

    # Test case 3: None as prompt
    print("Testing case [3/3] started.")
    prompt = None
    try:
        result_image = generate_image_from_text(prompt)
        assert False, "Test case [3/3] failed: Expected ValueError for None prompt."
    except ValueError as e:
        assert str(e) == 'The prompt cannot be empty.', f"Test case [3/3] failed: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_generate_image_from_text()