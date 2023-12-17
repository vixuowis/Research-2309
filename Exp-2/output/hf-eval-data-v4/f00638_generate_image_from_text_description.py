# requirements_file --------------------

!pip install -U diffusers transformers accelerate scipy safetensors

# function_import --------------------

from diffusers import StableDiffusionInpaintPipeline

# function_code --------------------

def generate_image_from_text_description(prompt, image_path=None, mask_path=None):
    # Instantiate the pipeline for stable diffusion inpainting
    pipe = StableDiffusionInpaintPipeline.from_pretrained('stabilityai/stable-diffusion-2-inpainting', torch_dtype=torch.float16)
    # Move pipeline to GPU for faster generation
    pipe = pipe.to('cuda')
    # Load the image and mask if provided
    image, mask_image = None, None
    if image_path and mask_path:
        image = Image.open(image_path)
        mask_image = Image.open(mask_path)
    # Generate the image based on the provided text prompt
    output_image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
    # Return the generated image object
    return output_image

# test_function_code --------------------

def test_generate_image_from_text_description():
    print('Testing generate_image_from_text_description function with a sample prompt.')
    sample_prompt = 'A cute cat sleeping on a couch.'
    generated_image = generate_image_from_text_description(sample_prompt)
    print('Test completed. Check the generated image object.')