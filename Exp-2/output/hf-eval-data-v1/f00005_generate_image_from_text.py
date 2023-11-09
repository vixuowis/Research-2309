from diffusers import StableDiffusionInpaintPipeline
import torch


def generate_image_from_text(prompt: str, output_file: str = 'generated_image.png'):
    """
    This function generates an image based on the provided text prompt using the StableDiffusionInpaintPipeline from Hugging Face.

    Parameters:
    prompt (str): The text description of the desired image.
    output_file (str): The name of the file to save the generated image. Default is 'generated_image.png'.
    """
    # Initialize the pipeline with the pre-trained model
    pipe = StableDiffusionInpaintPipeline.from_pretrained('stabilityai/stable-diffusion-2-inpainting', torch_dtype=torch.float16)
    pipe.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Generate the image based on the text prompt
    generated_image = pipe(prompt=prompt).images[0]

    # Save the generated image to a file
    generated_image.save(output_file)