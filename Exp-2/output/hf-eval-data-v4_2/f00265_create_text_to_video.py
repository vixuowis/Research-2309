# requirements_file --------------------

pip install torch diffusers

# function_import --------------------

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# function_code --------------------

def create_text_to_video(prompt):
    """
    Generate a video from textual description using a pretrained text-to-video model.

    Args:
        prompt (str): The textual description to visualize in the video.

    Returns:
        str: Path to the saved video file.

    Raises:
        ValueError: If prompt is empty.
        RuntimeError: If the model fails to generate the video.
    """
    if not prompt:
        raise ValueError('Prompt is empty.')

    pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b', torch_dtype=torch.float16, variant='fp16')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    video_frames = pipe(prompt, num_inference_steps=25).frames
    video_path = export_to_video(video_frames, 'output_video.mp4')
    
    return video_path

# test_function_code --------------------

def test_create_text_to_video():
    print("Testing started.")

    prompt_empty = ""
    prompt_non_empty = "A couple sitting in a cafe laughing."

    # Test case 1: Empty prompt
    print("Testing case [1/2] started.")
    try:
        create_text_to_video(prompt_empty)
        assert False, "Test case [1/2] failed: Prompt is empty but no ValueError raised."
    except ValueError as e:
        assert str(e) == 'Prompt is empty.', f"Test case [1/2] failed: {str(e)}"

    # Test case 2: Non-empty prompt
    print("Testing case [2/2] started.")
    try:
        video_path = create_text_to_video(prompt_non_empty)
        assert video_path.endswith('.mp4'), f"Test case [2/2] failed: Video path is not ending with .mp4: {video_path}"
    except Exception as e:
        assert False, f"Test case [2/2] failed: {str(e)}"

    print("Testing finished.")

# call_test_function_line --------------------

test_create_text_to_video()