# function_import --------------------

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, export_to_video

# function_code --------------------

def generate_video_from_text(prompt: str, num_inference_steps: int = 25) -> str:
    '''
    Generate a video from a text description using a pre-trained model.

    Args:
        prompt: A string that describes the scene to be generated in the video.
        num_inference_steps: An integer that specifies the number of inference steps. Default is 25.

    Returns:
        A string that is the path to the generated video.

    Raises:
        ValueError: If the prompt is not a string.
        ValueError: If the num_inference_steps is not an integer.
    '''
    if not isinstance(prompt, str):
        raise ValueError('Prompt must be a string.')
    if not isinstance(num_inference_steps, int):
        raise ValueError('Number of inference steps must be an integer.')

    pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b-legacy', torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    video_frames = pipe(prompt, num_inference_steps=num_inference_steps).frames
    video_path = export_to_video(video_frames)
    return video_path

# test_function_code --------------------

def test_generate_video_from_text():
    '''
    Test the function generate_video_from_text.
    '''
    # Test with a normal case
    video_path = generate_video_from_text('A dog jumps over a fence')
    assert isinstance(video_path, str)

    # Test with a different number of inference steps
    video_path = generate_video_from_text('A cat climbs a tree', 30)
    assert isinstance(video_path, str)

    # Test with an invalid prompt
    try:
        generate_video_from_text(123)
    except ValueError:
        pass
    else:
        raise AssertionError('Expected a ValueError when the prompt is not a string.')

    # Test with an invalid number of inference steps
    try:
        generate_video_from_text('A bird flies in the sky', 'twenty')
    except ValueError:
        pass
    else:
        raise AssertionError('Expected a ValueError when the number of inference steps is not an integer.')

    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_video_from_text()