from diffusers import StableDiffusionPipeline
import torch


def generate_space_party_image():
    """
    This function generates an image for a space party with astronauts and aliens having fun together.
    It uses the StableDiffusionPipeline from the diffusers package and a pre-trained model from Hugging Face.
    The generated image is saved locally as 'space_party.png'.
    """
    # Load the pre-trained model
    pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1', torch_dtype=torch.float16)

    # Define the text prompt
    prompt = "a space party with astronauts and aliens having fun together"

    # Generate the image
    image = pipe(prompt).images[0]

    # Save the image locally
    image.save('space_party.png')