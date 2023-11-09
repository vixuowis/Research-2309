from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from controlnet_aux import MLSDdetector
import torch


def generate_image(prompt: str, image_path: str, output_path: str):
    '''
    This function generates an image based on the given prompt using a pre-trained ControlNetModel.
    Args:
    prompt (str): The text prompt to generate the image from.
    image_path (str): The path to the base image.
    output_path (str): The path to save the generated image.
    '''
    # Load the pre-trained ControlNetModel
    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_mlsd', torch_dtype=torch.float16)
    # Load the image processor
    processor = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
    # Load the base image
    image = Image.open(image_path)
    # Process the image
    control_image = processor(image)
    # Create the pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16
    )
    # Generate the image
    generated_image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0), image=control_image).images[0]
    # Save the generated image
    generated_image.save(output_path)