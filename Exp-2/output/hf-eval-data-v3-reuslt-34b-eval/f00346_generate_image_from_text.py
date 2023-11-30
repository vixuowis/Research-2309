# function_import --------------------

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
import os

# function_code --------------------

def generate_image_from_text(prompt: str, model_id: str = 'stabilityai/stable-diffusion-2-1', save_path: str = 'generated_image.png'):
    """
    Generate an image based on the given text description using the StableDiffusionPipeline model.

    Args:
        prompt (str): The text description of the scene.
        model_id (str, optional): The id of the pre-trained model. Defaults to 'stabilityai/stable-diffusion-2-1'.
        save_path (str, optional): The path to save the generated image. Defaults to 'generated_image.png'.

    Returns:
        None
    """    

    # Set up device
    if torch.cuda.is_available():
      device = torch.device('cuda')
    else:
      device = torch.device('cpu') 

    # Load model
    model = StableDiffusionPipeline(model_id).to(device)

    # Create input embedding
    text = prompt
    emb = model.text_embedding(text, device=device)
    
    # Set up solver to generate an image
    timestep_respacing = 'fast27' 
    schedule = DPMSolverMultistepScheduler(diffusion_steps=(1000 - model.image_size) * 6 + 5, 
                                           model_mean_type='epsilon', noise_schedule="linear", 
                                           timestep_respacing=timestep_respacing, 
                                           use_kl=False)
    
    # Generate image
    output_image = model(emb, schedule=schedule)
    
    # Post-process image and save it to the given path
    output_image = (output_image.clamp(0., 1.) * 255).byte().cpu()
    os.makedirs('outputs', exist_ok=True)
    Image.fromarray(output_image[0, ...].numpy()).save(f'outputs/{save_path}')

# test_function_code --------------------

def test_generate_image_from_text():
    """
    Test the function generate_image_from_text.

    Returns:
        str: 'All Tests Passed' if all tests pass, otherwise the error message.
    """
    try:
        generate_image_from_text('a scene of a magical forest with fairies and elves')
        assert os.path.exists('generated_image.png')
        os.remove('generated_image.png')
        return 'All Tests Passed'
    except Exception as e:
        return str(e)


# call_test_function_code --------------------

print(test_generate_image_from_text())