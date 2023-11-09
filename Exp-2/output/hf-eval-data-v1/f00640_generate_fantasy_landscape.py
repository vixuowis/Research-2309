from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch


def generate_fantasy_landscape(prompt):
    '''
    This function generates an image of a fantasy landscape based on the provided text description.
    
    Parameters:
    prompt (str): The text description of the fantasy landscape.
    
    Returns:
    str: The path to the generated image.
    '''
    # Define the model ID
    model_id = 'darkstorm2150/Protogen_v5.8_Official_Release'
    
    # Load the Protogen x5.8 model
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    
    # Set the pipe scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Move the model to GPU for faster processing
    pipe = pipe.to('cuda')
    
    # Generate the image
    image_result = pipe(prompt, num_inference_steps=25).images[0]
    
    # Save the image to a file
    image_path = './result.jpg'
    image_result.save(image_path)
    
    return image_path