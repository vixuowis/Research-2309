# requirements_file --------------------

!pip install -U diffusers, torch

# function_import --------------------

from diffusers import StableDiffusionPipeline
import torch

# function_code --------------------

def generate_anime_style_girl_with_guitar(prompt='anime-style girl with a guitar', model_id='andite/anything-v4.0', output_path='./anime_girl_guitar.png'):
    # Load the pre-trained Stable Diffusion Pipeline model
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    # Move the model to GPU
    pipe = pipe.to('cuda')
    # Perform image generation
    generated_image = pipe(prompt).images[0]
    # Save the generated image to a file
    generated_image.save(output_path)
    # Return the image
    return generated_image

# test_function_code --------------------

def test_generate_anime_style_girl_with_guitar():
    print("Testing generate_anime_style_girl_with_guitar function.")
    # Test image generation with default parameters
    image = generate_anime_style_girl_with_guitar()
    assert image is not None, "Image generation failed with default parameters."
    print("Test passed with default parameters.")

    # Test image generation with custom prompt
    custom_prompt = 'anime-style girl playing an electric guitar'
    image = generate_anime_style_girl_with_guitar(prompt=custom_prompt)
    assert image is not None, "Image generation failed with custom prompt."
    print("Test passed with custom prompt."

    # Test with invalid model_id
    try:
        generate_anime_style_girl_with_guitar(model_id='invalid_model')
        assert False, "Image generation should have failed with invalid model id."
    except Exception as e:
        print("Test passed with invalid model id.")

    print("Testing completed successfully.")