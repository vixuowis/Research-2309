# function_import --------------------

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# function_code --------------------

def generate_video(prompt: str, num_inference_steps: int = 25) -> str:
    """
    Generate a video based on the provided text prompt using a pre-trained text-to-video model.

    Args:
        prompt (str): The text prompt based on which the video is to be generated.
        num_inference_steps (int, optional): The number of inference steps. Default is 25.

    Returns:
        str: The path to the generated video.
    """
    pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b-legacy', torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    video_frames = pipe(prompt, num_inference_steps=num_inference_steps).frames
    video_path = export_to_video(video_frames)
    return video_path

# test_function_code --------------------

def test_generate_video():
    """
    Test the generate_video function.
    """
    prompt = 'Spiderman is surfing'
    video_path = generate_video(prompt)
    assert isinstance(video_path, str) and video_path.endswith('.mp4')

# call_test_function_code --------------------

test_generate_video()