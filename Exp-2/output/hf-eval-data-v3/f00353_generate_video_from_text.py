# function_import --------------------

import torch
from diffusers import DiffusionPipeline
from diffusers.schedulers_async import DPMSolverMultistepScheduler

# function_code --------------------

def generate_video_from_text(prompt: str, num_inference_steps: int = 25):
    '''
    Generate video from text using a pre-trained text-to-video diffusion model.

    Args:
        prompt (str): The description of the video to be generated.
        num_inference_steps (int, optional): The number of inference steps. Default is 25.

    Returns:
        video_frames (torch.Tensor): The generated video frames.

    Raises:
        ModuleNotFoundError: If the required modules are not found.
    '''
    pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b', torch_dtype=torch.float16, variant='fp16')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    video_frames = pipe(prompt, num_inference_steps=num_inference_steps).frames
    return video_frames

# test_function_code --------------------

def test_generate_video_from_text():
    '''
    Test the function generate_video_from_text.
    '''
    prompt = 'a person walking along a beach'
    video_frames = generate_video_from_text(prompt)
    assert video_frames is not None, 'The generated video frames should not be None.'
    assert video_frames.shape[0] > 0, 'The number of frames should be greater than 0.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_video_from_text()