# function_import --------------------

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import os

# function_code --------------------

def generate_image(prompt: str, model_id: str = 'stabilityai/stable-diffusion-2-1-base', output_file: str = 'output.png'):
    """
    Generate an image based on the provided text prompt using the StableDiffusionPipeline.

    Args:
        prompt (str): The text prompt to generate the image from.
        model_id (str, optional): The model id to use for the generation. Defaults to 'stabilityai/stable-diffusion-2-1-base'.
        output_file (str, optional): The file to save the generated image to. Defaults to 'output.png'.

    Returns:
        None
    """    

    # Check if CUDA is available and set device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model with the specified id from the hub
    model, options = StableDiffusionPipeline.from_hub(model_id)

    # Move the model to the selected device and load it into eval mode (necessary for this type of model)
    model.to(device).eval()

    # Create a sample context dictionary using the provided prompt string
    context = {"caption": prompt}

    # Initialize an EulerDiscreteScheduler with our options dictionary and use that to create a sampling loop
    scheduler = EulerDiscreteScheduler(options)
    
    # Sample the model using the scheduler we just initialized.
    # We don't use any labels since this is an unconditional sample
    samples, intermediates = scheduler.sample_with_halting_criterion(model=model, options=options, context=context)

    # Get the final sample out of the returned list of samples
    sample = samples[-1]
    
    # Convert the generated image to RGB and save it to the output file.
    img = Image.fromarray(sample['image'][0].astype(np.uint8))
    img.convert('RGB').save(output_file)

# test_function_code --------------------

def test_generate_image():
    """
    Test the generate_image function.
    """
    generate_image('a lighthouse on a foggy island', output_file='test_output.png')
    assert os.path.exists('test_output.png'), 'Test failed: Image file not found.'
    os.remove('test_output.png')
    print('All Tests Passed')


# call_test_function_code --------------------

test_generate_image()