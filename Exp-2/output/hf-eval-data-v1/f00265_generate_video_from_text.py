import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

def generate_video_from_text(prompt):
    '''
    This function generates a video from a given text prompt using a pretrained model from Hugging Face.
    The model is based on a multi-stage text-to-video generation diffusion model.
    
    Args:
    prompt (str): The text prompt to generate the video from.
    
    Returns:
    str: The path to the generated video.
    '''
    # Load the pretrained model
    pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b', torch_dtype=torch.float16, variant='fp16')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    
    # Generate the video frames from the prompt
    video_frames = pipe(prompt, num_inference_steps=25).frames
    
    # Export the video frames to a video file
    video_path = export_to_video(video_frames, 'output_video.mp4')
    
    return video_path