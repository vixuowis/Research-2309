# requirements_file --------------------

!pip install -U diffusers transformers scipy torch

# function_import --------------------

import torch
from diffusers import StableDiffusionPipeline


# function_code --------------------

def generate_3d_print_design(prompt):
    model_id = 'CompVis/stable-diffusion-v1-4'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    image = pipe(prompt).images[0]
    image.save(f'{prompt.replace(' ', '_')}_3d_print_design.png')
    return image


# test_function_code --------------------

def test_generate_3d_print_design():
    print("Testing started.")

    prompt = "a futuristic 3D printed car"
    print("Testing generate_3d_print_design function.")
    image = generate_3d_print_design(prompt)
    assert image is not None, "Test failed: The function did not return an image."
    assert image.format == 'PNG', "Test failed: The image format is not PNG."

    print("Testing finished.")

# Run the test function
test_generate_3d_print_design()
