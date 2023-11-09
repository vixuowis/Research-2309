import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, export_to_video


def generate_video_from_text(prompt: str, num_inference_steps: int = 25) -> str:
    """
    This function generates a video from a given text description using a pre-trained model from Hugging Face.
    
    Args:
        prompt (str): The text description to generate the video from.
        num_inference_steps (int, optional): The number of inference steps. Defaults to 25.
    
    Returns:
        str: The path to the generated video.
    """
    # Instantiate the DiffusionPipeline with the pre-trained model
    pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b-legacy', torch_dtype=torch.float16)
    
    # Configure the scheduler using the loaded model's configuration
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Enable CPU offloading to save GPU memory
    pipe.enable_model_cpu_offload()
    
    # Generate the video frames
    video_frames = pipe(prompt, num_inference_steps=num_inference_steps).frames
    
    # Export the video frames to a video file
    video_path = export_to_video(video_frames)
    
    return video_path