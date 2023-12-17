# requirements_file --------------------

pip install diffusers transformers accelerate

# function_import --------------------

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# function_code --------------------

def generate_video_from_text_description(prompt: str) -> str:
    """
    Generates a video from a given text description using a text-to-video API.

    Args:
        prompt (str): A text description of the video content.

    Returns:
        str: The path to the generated video file.

    Raises:
        RuntimeError: If there is an error loading the model or generating the video.
    """
    try:
        pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b', torch_dtype=torch.float16, variant='fp16')
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        video_frames = pipe(prompt, num_inference_steps=25).frames
        return export_to_video(video_frames)
    except Exception as e:
        raise RuntimeError('Error in generating video') from e

# test_function_code --------------------

def test_generate_video_from_text_description():
    print("Testing started.")
    # No dataset to load, so using a hardcoded example

    # Example text description
    prompt = "cats playing with laser pointer"

    # Testing case 1: Check if a string path is returned
    print("Testing case [1/1] started.")
    video_path = generate_video_from_text_description(prompt)
    assert isinstance(video_path, str), f"Test case [1/1] failed: Expected string path, got {type(video_path)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_video_from_text_description()