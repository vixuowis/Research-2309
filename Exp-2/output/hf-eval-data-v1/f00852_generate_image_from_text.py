import torch
from controlnet_aux import OpenposeDetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

def generate_image_from_text(input_image: torch.Tensor, text_prompt: str, num_inference_steps: int = 30, seed: int = 0) -> torch.Tensor:
    """
    Generate an image based on a textual description using a pretrained ControlNetModel.

    Args:
        input_image (torch.Tensor): The input image to process.
        text_prompt (str): The textual description of the scene.
        num_inference_steps (int, optional): The number of inference steps. Defaults to 30.
        seed (int, optional): The seed for the random number generator. Defaults to 0.

    Returns:
        torch.Tensor: The generated image.
    """
    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_openpose', torch_dtype=torch.float16)
    openpose_detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    control_image = openpose_detector(input_image, hand_and_face=True)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    generator = torch.manual_seed(seed)
    output_image = pipe(text_prompt, num_inference_steps=num_inference_steps, generator=generator, image=control_image).images[0]
    return output_image