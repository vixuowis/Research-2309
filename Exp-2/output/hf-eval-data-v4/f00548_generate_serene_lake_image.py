# requirements_file --------------------

!pip install -U diffusers transformers scipy

# function_import --------------------

import torch
from diffusers import StableDiffusionPipeline

# function_code --------------------

def generate_serene_lake_image(prompt):
    """
    Generates an image of a serene lake at sunset based on the given text prompt.

    :param prompt: Text description of the image to be generated.
    :return: Generated image object.
    """
    model_id = 'CompVis/stable-diffusion-v1-4'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    # Generate the image
    image = pipe(prompt).images[0]

    return image

# test_function_code --------------------

def test_generate_serene_lake_image():
    print("Testing started.")

    # Test with the given prompt description
    prompt = 'a serene lake at sunset'

    # Test case: Generating image
    print("Testing case [1/1] started.")
    image = generate_serene_lake_image(prompt)
    assert image is not None, "Test case failed: The generated image is None."    print("Testing finished.")

# Run test
print("\nRunning test for generate_serene_lake_image function")
test_generate_serene_lake_image()