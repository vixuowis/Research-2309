import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video


def generate_spiderman_surfing_video():
    """
    This function generates a video of Spiderman surfing using a pre-trained text-to-video model.
    The model is loaded from Hugging Face's model hub.
    """
    # Load the pre-trained text-to-video model
    pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b-legacy', torch_dtype=torch.float16)
    
    # Set the scheduler of the diffusion model
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Offload the model to CPU
    pipe.enable_model_cpu_offload()
    
    # Provide the prompt for the video generation
    prompt = "Spiderman is surfing"
    
    # Generate the video frames
    video_frames = pipe(prompt, num_inference_steps=25).frames
    
    # Export the frames to a video file
    video_path = export_to_video(video_frames)
    
    return video_path