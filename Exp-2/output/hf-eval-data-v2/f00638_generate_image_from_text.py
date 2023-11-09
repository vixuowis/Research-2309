# function_import --------------------

from diffusers import StableDiffusionInpaintPipeline

# function_code --------------------

def generate_image_from_text(prompt: str, image=None, mask_image=None):
    """
    Generate an image based on a text description using the StableDiffusionInpaintPipeline model.

    Args:
        prompt (str): The text description to generate the image from.
        image: The initial image to modify. If None, a new image will be generated.
        mask_image: The mask image to apply. If None, no mask will be applied.

    Returns:
        The generated image.
    """
    pipe = StableDiffusionInpaintPipeline.from_pretrained('stabilityai/stable-diffusion-2-inpainting', torch_dtype=torch.float16)
    pipe.to('cuda')
    output_image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
    return output_image

# test_function_code --------------------

def test_generate_image_from_text():
    """
    Test the generate_image_from_text function.
    """
    prompt = 'A beautiful landscape with a waterfall and a sunset'
    output_image = generate_image_from_text(prompt)
    assert output_image is not None, 'The generated image should not be None.'

# call_test_function_code --------------------

test_generate_image_from_text()