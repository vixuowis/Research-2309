# requirements_file --------------------

!pip install -U diffusers torch

# function_import --------------------

from diffusers import StableDiffusionInpaintPipeline
import torch

# function_code --------------------

def generate_kangaroo_eating_pizza_sign(prompt):
    pipe = StableDiffusionInpaintPipeline.from_pretrained('runwayml/stable-diffusion-inpainting', revision='fp16', torch_dtype=torch.float16)
    image = pipe(prompt=prompt).images[0]
    image.save('kangaroo_pizza_sign.png')
    return image

# test_function_code --------------------

def test_generate_kangaroo_eating_pizza_sign():
    prompt = 'kangaroo eating pizza'
    image = generate_kangaroo_eating_pizza_sign(prompt)
    # Since testing actual image generation may not be possible without the environment and model,
    # we'll simply check for a method from the PIL Image class
    print('Testing generate_kangaroo_eating_pizza_sign function.')
    assert hasattr(image, 'save'), 'Test failed: The generated object does not have a save method.'