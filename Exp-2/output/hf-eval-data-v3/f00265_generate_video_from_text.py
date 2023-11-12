# function_import --------------------

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# function_code --------------------

def generate_video_from_text(prompt: str, model_name: str = 'damo-vilab/text-to-video-ms-1.7b', torch_dtype: str = 'torch.float16', variant: str = 'fp16', num_inference_steps: int = 25, output_file: str = 'output_video.mp4') -> str:
    """
    Generate a video from a text prompt using a pretrained model.

    Args:
        prompt (str): The text prompt to generate the video from.
        model_name (str, optional): The name of the pretrained model to use. Defaults to 'damo-vilab/text-to-video-ms-1.7b'.
        torch_dtype (str, optional): The data type to use for torch tensors. Defaults to 'torch.float16'.
        variant (str, optional): The variant of the model to use. Defaults to 'fp16'.
        num_inference_steps (int, optional): The number of inference steps to use. Defaults to 25.
        output_file (str, optional): The path to the output video file. Defaults to 'output_video.mp4'.

    Returns:
        str: The path to the output video file.
    """
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype, variant=variant)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    video_frames = pipe(prompt, num_inference_steps=num_inference_steps).frames
    video_path = export_to_video(video_frames, output_file)

    return video_path

# test_function_code --------------------

def test_generate_video_from_text():
    """
    Test the generate_video_from_text function.
    """
    prompt = 'A couple sitting in a cafe and laughing while using our product'
    output_file = 'test_output_video.mp4'

    # Test default parameters
    assert generate_video_from_text(prompt) == 'output_video.mp4'

    # Test custom parameters
    assert generate_video_from_text(prompt, output_file=output_file) == output_file

    # Test invalid prompt
    try:
        generate_video_from_text('')
    except Exception as e:
        assert isinstance(e, ValueError)

    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_generate_video_from_text())