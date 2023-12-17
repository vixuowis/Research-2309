# requirements_file --------------------

!pip install -U diffusers transformers accelerate scipy safetensors

# function_import --------------------

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image

# function_code --------------------

def generate_storybook_image(prompt):
    # Import necessary libraries
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    import torch
    from PIL import Image
    
    # Initialize the pipeline with the pre-trained model
    model_id = 'stabilityai/stable-diffusion-2-1'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Move the pipeline to the GPU for faster inference
    pipe = pipe.to('cuda')
    
    # Generate the image with the provided text prompt
    generated_image = pipe(prompt).images[0]
    
    # Save the generated image
    image_filename = prompt.replace(' ', '_') + '.png'
    generated_image.save(image_filename)
    
    return image_filename

# test_function_code --------------------

def test_generate_storybook_image():
    print('Testing generate_storybook_image function.')
    
    # Test case: Generate an image with a given prompt
    prompt = 'a magical forest with fairies and elves'
    image_filename = generate_storybook_image(prompt)
    assert image_filename == 'a_magical_forest_with_fairies_and_elves.png', f'Test failed: Expected filename a_magical_forest_with_fairies_and_elves.png, Got {image_filename}'
    
    # Additional test cases could include different prompts and the corresponding expected filenames
    
    print('All tests passed!')

# Run the test function
test_generate_storybook_image()