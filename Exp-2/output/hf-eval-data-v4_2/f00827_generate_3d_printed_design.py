# requirements_file --------------------

!pip install -U diffusers transformers scipy

# function_import --------------------

import torch
from diffusers import StableDiffusionPipeline

# function_code --------------------

def generate_3d_printed_design(prompt: str) -> str:
    """
    Generates an image based on the provided text prompt using the Stable Diffusion model.

    Args:
        prompt (str): A text description of the image to be generated.

    Returns:
        str: The filename of the saved image.

    Raises:
        RuntimeError: If the device is not configured correctly or model loading fails.
    """
    model_id = 'CompVis/stable-diffusion-v1-4'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        image = pipe(prompt).images[0]
        filename = prompt.replace(' ', '_') + '.png'
        image.save(filename)
        return filename
    except Exception as e:
        raise RuntimeError('Failed to generate image: ' + str(e))

# test_function_code --------------------

def test_generate_3d_printed_design():
    print("Testing started.")
    sample_prompts = [
        'a futuristic 3D printed car',
        'a detailed 3D model of a dragon',
        'a 3D printable design of a modern house'
    ]

    for i, prompt in enumerate(sample_prompts):
        print(f"Testing case [{i+1}/{len(sample_prompts)}] started.")
        filename = generate_3d_printed_design(prompt)
        assert filename.endswith('.png'), f"Test case [{i+1}/{len(sample_prompts)}] failed: Expected .png extension"
        print(f"Testing case [{i+1}/{len(sample_prompts)}] succeeded.")
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_3d_printed_design()