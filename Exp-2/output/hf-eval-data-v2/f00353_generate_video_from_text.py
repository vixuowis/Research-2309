# function_import --------------------

from diffusers import DiffusionPipeline
from diffusers.schedulers_async import DPMSolverMultistepScheduler
import torch

# function_code --------------------

def generate_video_from_text(prompt: str, num_inference_steps: int = 25):
    """
    Generate video from a given text description using a pre-trained text-to-video diffusion model.

    Args:
        prompt (str): The text description to generate the video from.
        num_inference_steps (int, optional): The number of inference steps to take. Defaults to 25.

    Returns:
        video_frames: The generated video frames.
    """
    pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b', torch_dtype=torch.float16, variant='fp16')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    video_frames = pipe(prompt, num_inference_steps=num_inference_steps).frames
    return video_frames

# test_function_code --------------------

def test_generate_video_from_text():
    """
    Test the 'generate_video_from_text' function.
    """
    prompt = 'a person walking along a beach'
    video_frames = generate_video_from_text(prompt)
    assert video_frames is not None, 'No video frames were generated.'
    assert len(video_frames) > 0, 'No video frames were generated.'

# call_test_function_code --------------------

test_generate_video_from_text()