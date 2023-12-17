# requirements_file --------------------

!pip install -U diffusers transformers accelerate scipy safetensors

# function_import --------------------

from diffusers import StableDiffusionInpaintPipeline
import torch

# function_code --------------------

def generate_living_room_image(prompt):
    # Initialize the diffusion pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained('stabilityai/stable-diffusion-2-inpainting', torch_dtype=torch.float16)
    pipe.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Generate an image based on the given prompt
    generated_image = pipe(prompt=prompt).images[0]

    # Return the image object
    return generated_image

# test_function_code --------------------

def test_generate_living_room_image():
    print("Testing started.")
    prompt = "A modern living room with a fireplace and a large window overlooking a forest."

    # Generate image from the prompt
    generated_image = generate_living_room_image(prompt)

    # Test case: Check if an image was returned
    print("Testing image generation.")
    assert generated_image is not None, "Test failed: No image was returned."

    print("Testing finished.")

# Run the test function
test_generate_living_room_image()