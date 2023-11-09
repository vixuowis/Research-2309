import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

def generate_video_from_text(prompt):
    '''
    This function generates a video from a given text using a pre-trained model from Hugging Face.
    The model is based on a multi-stage text-to-video generation diffusion model.
    
    Parameters:
    prompt (str): The text based on which the video is to be generated.
    
    Returns:
    str: The path where the generated video is saved.
    '''
    # Load the pre-trained model
    pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b', torch_dtype=torch.float16, variant='fp16')
    # Set the multi-step scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # Enable CPU offloading to save GPU memory
    pipe.enable_model_cpu_offload()
    # Generate video frames based on the prompt
    video_frames = pipe(prompt, num_inference_steps=25).frames
    # Export the generated video frames to a video and return the path
    return export_to_video(video_frames)