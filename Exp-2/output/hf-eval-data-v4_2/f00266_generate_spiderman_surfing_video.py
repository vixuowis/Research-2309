# requirements_file --------------------

!pip install -U torch diffusers transformers accelerate

# function_import --------------------

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# function_code --------------------

def generate_spiderman_surfing_video(prompt: str, num_inference_steps: int = 25) -> str:
    """
    Generate a video of Spiderman surfing based on the provided prompt using a text-to-video diffusion model.

    Args:
        prompt (str): The text prompt to describe the video content. Should be a description of Spiderman surfing.
        num_inference_steps (int, optional): The number of steps for the diffusion process. Defaults to 25.

    Returns:
        str: The path to the generated video file.

    Raises:
        ValueError: If the prompt is empty.
    """
    # Validate the prompt
    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    # Load the pre-trained model
    pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b-legacy', torch_dtype=torch.float16)
    # Set the scheduler and offload model to CPU
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # Generate the video frames
    video_frames = pipe(prompt, num_inference_steps=num_inference_steps).frames
    # Export frames to a video file
    video_path = export_to_video(video_frames)

    return video_path

# test_function_code --------------------

def test_generate_spiderman_surfing_video():
    print("Testing started.")

    # Test case 1: Valid prompt
    print("Testing case [1/1] started.")
    video_path = generate_spiderman_surfing_video("Spiderman is surfing")
    assert video_path.endswith('.mp4'), f"Test case failed: Expected video file path to have .mp4 extension, got {video_path}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_spiderman_surfing_video()