import torch
from pathlib import Path
from diffusers.utils import load_image
from controlnet_aux import PidiNetDetector, HEDdetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler


def generate_image(prompt):
    '''
    This function generates an image based on a text prompt using the ControlNetModel from Hugging Face.
    Args:
    prompt (str): The text prompt to generate the image from.
    Returns:
    str: The path to the generated image.
    '''
    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_softedge', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    generator = torch.manual_seed(0)
    generated_image = pipe(prompt, num_inference_steps=30, generator=generator).images[0]
    generated_image.save('generated_image.png')
    return 'generated_image.png'