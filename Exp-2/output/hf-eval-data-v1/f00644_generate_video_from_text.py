import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

def generate_video_from_text(prompt):
    '''
    This function generates a video from a text description using the damo-vilab/text-to-video-ms-1.7b model from Hugging Face.
    
    Args:
    prompt (str): The text description to generate the video from.
    
    Returns:
    str: The path to the generated video.
    '''
    # Load the model
    pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b', torch_dtype=torch.float16, variant='fp16')
    # Configure the scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # Enable CPU offload for the model
    pipe.enable_model_cpu_offload()
    # Generate the video frames from the text description
    video_frames = pipe(prompt, num_inference_steps=25).frames
    # Export the video frames to a video file and return the path to the file
    return export_to_video(video_frames)