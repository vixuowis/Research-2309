# requirements_file --------------------

!pip install -U diffusers transformers accelerate scipy safetensors

# function_import --------------------

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

# function_code --------------------

def generate_lighthouse_image(prompt='a lighthouse on a foggy island'):
    # Install required dependencies
    # pip install diffusers transformers accelerate scipy safetensors

    # Pretrained model and scheduler from Hugging Face
    model_id = 'stabilityai/stable-diffusion-2-1-base'
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder='scheduler')

    # Initialize Stable Diffusion Pipeline with the model and scheduler
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to('cuda')

    # Generate the image based on the prompt
    output = pipe(prompt)
    image = output.images[0]

    # Save the image
    image.save('lighthouse_foggy_island.png')
    return image

# test_function_code --------------------

def test_generate_lighthouse_image():
    print("Testing started.")

    # Test case: Generate image with default prompt
    print("Testing default prompt image generation.")
    default_image = generate_lighthouse_image()
    assert default_image is not None, 'Default prompt image generation failed.'

    # Test case: Validate image format
    print("Validating image format.")
    assert isinstance(default_image, Image.Image), 'Generated image is not in PIL image format.'

    # Test case: Generate image with custom prompt
    print("Generating image with custom prompt.")
    custom_prompt = 'a lighthouse surrounded by the stormy sea'
    custom_image = generate_lighthouse_image(custom_prompt)
    assert custom_image is not None, "Custom prompt ('" + custom_prompt + "') image generation failed."

    print("Testing finished.")

# Run the test function
test_generate_lighthouse_image()