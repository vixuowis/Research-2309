# requirements_file --------------------

!pip install -U diffusers transformers accelerate scipy safetensors

# function_import --------------------

from diffusers import StableDiffusionInpaintPipeline
import torch

# function_code --------------------

def generate_modern_living_room_image(prompt: str, model_name: str = 'stabilityai/stable-diffusion-2-inpainting', output_file: str = 'modern_living_room.png') -> str:
    """
    Generate an image of a modern living room with the given text prompt using Stable Diffusion Inpainting model.

    Args:
        prompt (str): A description of the image to generate.
        model_name (str): The name of the pre-trained Stable Diffusion model to use. Defaults to 'stabilityai/stable-diffusion-2-inpainting'.
        output_file (str): The path where the generated image will be saved. Defaults to 'modern_living_room.png'.

    Returns:
        str: The path to the saved image.

    Raises:
        Exception: If image generation fails.

    """
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        pipe.to(device)

        generated_image = pipe(prompt=prompt).images[0]
        generated_image.save(output_file)

        return output_file
    except Exception as e:
        raise Exception(f'Image generation failed: {e}')

# test_function_code --------------------

def test_generate_modern_living_room_image():
    print("Testing started.")
    prompt = "A modern living room with a fireplace and a large window overlooking a forest."

    # Testing case 1: Normal case
    print("Testing case [1/1] started.")
    try:
        output_file = generate_modern_living_room_image(prompt)
        assert output_file == 'modern_living_room.png', f"Test case [1/1] failed: Expected 'modern_living_room.png', got {output_file}"
    except Exception as e:
        assert False, f"Test case [1/1] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_modern_living_room_image()