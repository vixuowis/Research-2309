import torch
from controlnet_aux import MLSDdetector
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

def generate_toy_robot_image(prompt: str, initial_image=None):
    '''
    This function generates an image of a toy robot using a pretrained ControlNet model.
    Args:
    prompt (str): The text prompt to control the image generation.
    initial_image (str, optional): The path to an initial image. Defaults to None.
    Returns:
    str: The path to the generated image.
    '''
    processor = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
    control_image = processor(initial_image)
    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_mlsd', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    generator = torch.manual_seed(0)
    image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    image_path = "images/toy_robot.png"
    image.save(image_path)
    return image_path