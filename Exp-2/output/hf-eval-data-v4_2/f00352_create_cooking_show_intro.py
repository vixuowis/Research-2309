# requirements_file --------------------

pip install diffusers transformers accelerate

# function_import --------------------

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# function_code --------------------

def create_cooking_show_intro(prompt_text):
    """
    Generates a video for a cooking show intro based on the given text prompt.

    Args:
        prompt_text (str): The text prompt to generate the video from.

    Returns:
        str: The file path to the generated video.

    Raises:
        ValueError: If prompt_text is empty.
    """
    if not prompt_text:
        raise ValueError('The prompt_text cannot be empty.')

    pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b', torch_dtype=torch.float16, variant='fp16')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    video_frames = pipe(prompt_text, num_inference_steps=25).frames
    video_path = export_to_video(video_frames)
    return video_path

# test_function_code --------------------

def test_create_cooking_show_intro():
    print("Testing started.")
    sample_prompt = 'Chef John\'s Culinary Adventures'

    # Test case 1: Valid prompt
    print("Testing case [1/1] started.")
    assert create_cooking_show_intro(sample_prompt), f"Test case [1/1] failed: Expected a valid video path, got None or empty"
    print("Testing finished.")

# call_test_function_line --------------------

test_create_cooking_show_intro()