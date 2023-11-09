from diffusers import StableDiffusionPipeline
import torch


def generate_image(prompt):
    '''
    This function generates a high-resolution image based on the provided textual prompt using the StableDiffusionPipeline model from Hugging Face.
    
    Parameters:
    prompt (str): The textual description of the image to be generated.
    
    Returns:
    None
    '''
    # Define the model ID
    model_id = 'prompthero/openjourney'
    
    # Create an instance of the StableDiffusionPipeline and load the pre-trained model
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    
    # Move the model to the GPU
    pipe = pipe.to('cuda')
    
    # Generate the image based on the provided prompt
    image = pipe(prompt).images[0]
    
    # Save the generated image
    image.save('./vintage_sports_car_desert_sunset.png')