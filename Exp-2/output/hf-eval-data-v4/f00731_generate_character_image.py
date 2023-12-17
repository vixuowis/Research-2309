# requirements_file --------------------

!pip install -U diffusers, torch

# function_import --------------------

from diffusers import StableDiffusionPipeline
import torch

# function_code --------------------

def generate_character_image(prompt, negative_prompt):
    model_id = 'dreamlike-art/dreamlike-anime-1.0'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to('cuda')
    image = pipe(prompt, negative_prompt=negative_prompt).images[0]
    return image

# test_function_code --------------------

def test_generate_character_image():
    print("Testing generate_character_image function.")
    prompt = 'a brave warrior with a sword'
    negative_prompt = 'blurry, lowres'
    image = generate_character_image(prompt, negative_prompt)
    assert image is not None, "Image generation failed."
    print("Test case passed.")

test_generate_character_image()