# requirements_file --------------------

import subprocess

requirements = ["torch", "tuneavideo"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid
import torch

# function_code --------------------

def generate_redshift_video(prompt, video_length=8, height=512, width=512, num_inference_steps=50, guidance_scale=7.5):
    """
    Generates a video based on the input prompt in redshift style using a pretrained text-to-video pipeline.

    Args:
        prompt (str): Textual description of the video to generate.
        video_length (int, optional): The length of the video in frames. Defaults to 8.
        height (int, optional): The height of the video in pixels. Defaults to 512.
        width (int, optional): The width of the video in pixels. Defaults to 512.
        num_inference_steps (int, optional): The number of steps for the generation process. Defaults to 50.
        guidance_scale (float, optional): The scale for guiding the generation towards the prompt. Defaults to 7.5.

    Returns:
        str: Path to the saved video file.

    Raises:
        RuntimeError: If the generation process fails.
    """
    # Load the necessary models
    unet = UNet3DConditionModel.from_pretrained('Tune-A-Video-library/redshift-man-skiing', subfolder='unet', torch_dtype=torch.float16).to('cuda')
    pipe = TuneAVideoPipeline.from_pretrained('nitrosocke/redshift-diffusion', unet=unet, torch_dtype=torch.float16).to('cuda')
    # Enable memory efficient attention if required
    pipe.enable_xformers_memory_efficient_attention()
    # Generate video
    video = pipe(prompt, video_length=video_length, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).videos
    # Save the video
    video_path = f'./{prompt}.gif'
    save_videos_grid(video, video_path)
    # Return path to the saved video
    return video_path

# test_function_code --------------------

def test_generate_redshift_video():
    print("Testing started.")
    # Assume a mock function 'mock_prompt' simulates the prompting
    mock_prompt = '(redshift style) testing video generation'

    # Testing case 1: Prompt provided
    print("Testing case [1/1] started.")
    video_path = generate_redshift_video(mock_prompt)
    assert type(video_path) == str and video_path.endswith('.gif'), f"Test case [1/1] failed: Expected a string path ending with .gif, got {video_path}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_redshift_video()