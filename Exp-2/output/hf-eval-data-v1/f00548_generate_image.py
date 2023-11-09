import torch
from diffusers import StableDiffusionPipeline


def generate_image(prompt):
    '''
    This function generates an image based on the given text prompt using the StableDiffusionPipeline from Hugging Face.
    
    Parameters:
    prompt (str): The text description of the image to be generated.
    
    Returns:
    None
    '''
    # Load the pretrained model 'CompVis/stable-diffusion-v1-4' from Hugging Face's model hub
    model_id = 'CompVis/stable-diffusion-v1-4'
    # Set the device to the GPU if it is available for faster processing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize the pipeline with the pretrained model
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    # Generate the image based on the text prompt
    image = pipe(prompt).images[0]
    # Save the resulting image to your desired location
    image.save('generated_image.png')