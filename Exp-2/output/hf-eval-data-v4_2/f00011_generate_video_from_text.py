# requirements_file --------------------

!pip install -U diffusers transformers accelerate

# function_import --------------------

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# function_code --------------------

def generate_video_from_text(prompt, num_inference_steps=25):
    """
    Generate a video from a given text description.

    Args:
        prompt (str): Text description of the story or scene.
        num_inference_steps (int): The number of inference steps for the diffusion model.

    Returns:
        str: Path of the generated video file.

    Raises:
        ValueError: If prompt is not a string.
        RuntimeError: If generation process fails.
    """
    # Validate input prompt
    if not isinstance(prompt, str):
        raise ValueError('Prompt must be a string.')

    # Initialize the text-to-video generation pipeline
    pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b-legacy', torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # Generate video frames from the text prompt
    video_frames = pipe(prompt, num_inference_steps=num_inference_steps).frames

    # Export frames to a video file
    video_path = export_to_video(video_frames)

    # Return the path to the generated video file
    return video_path

# test_function_code --------------------

def test_generate_video_from_text():
    print("Testing started.")

    # Test case 1: Check that the function accepts string prompts
    print("Testing case [1/3] started.")
    try:
        video_path = generate_video_from_text('A dog jumps over a fence')
        assert isinstance(video_path, str), f"Test case [1/3] failed: Expected a string path, got {type(video_path)}"
    except ValueError as e:
        assert False, f"Test case [1/3] failed: {e}"

    # Test case 2: Check that the function raises a ValueError for non-string prompts
    print("Testing case [2/3] started.")
    try:
        generate_video_from_text(123)
        assert False, "Test case [2/3] failed: Expected ValueError for non-string prompt"
    except ValueError:
        pass

    # Test case 3: Check that the function raises an error for empty strings
    print("Testing case [3/3] started.")
    try:
        generate_video_from_text('')
        assert False, "Test case [3/3] failed: Expected error for empty prompt"
    except RuntimeError:
        pass

    print("Testing finished.")

# call_test_function_line --------------------

test_generate_video_from_text()