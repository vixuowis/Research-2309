from diffusers import DiffusionPipeline
from diffusers.schedulers_async import DPMSolverMultistepScheduler
import torch


def generate_video_from_text(prompt):
    """
    This function generates a video from a given text description using a pre-trained text-to-video diffusion model.
    
    Parameters:
    prompt (str): The text description to generate the video from.
    
    Returns:
    video_frames (list): The generated video frames.
    """
    # Load the pre-trained text-to-video diffusion model
    pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b', torch_dtype=torch.float16, variant='fp16')
    # Set the scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # Enable model CPU offload
    pipe.enable_model_cpu_offload()
    # Generate the video frames from the given text description
    video_frames = pipe(prompt, num_inference_steps=25).frames
    return video_frames