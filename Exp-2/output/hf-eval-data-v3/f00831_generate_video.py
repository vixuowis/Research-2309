# function_import --------------------

from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid
import torch
import os

# function_code --------------------

def generate_video(prompt, video_length, height, width, num_inference_steps, guidance_scale):
    """
    Generate a video based on a textual prompt using a pretrained model.

    Args:
        prompt (str): Textual description of the desired video.
        video_length (int): The length of the video to be generated.
        height (int): The height of the video to be generated.
        width (int): The width of the video to be generated.
        num_inference_steps (int): The number of inference steps to be performed.
        guidance_scale (float): The scale of guidance for the video generation.

    Returns:
        str: Path to the generated video file.
    """
    unet = UNet3DConditionModel.from_pretrained('Tune-A-Video-library/redshift-man-skiing', subfolder='unet', torch_dtype=torch.float16).to('cuda')
    pipe = TuneAVideoPipeline.from_pretrained('nitrosocke/redshift-diffusion', unet=unet, torch_dtype=torch.float16).to('cuda')
    pipe.enable_xformers_memory_efficient_attention()
    video = pipe(prompt, video_length=video_length, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).videos
    video_path = f'./{prompt}.gif'
    save_videos_grid(video, video_path)
    return video_path

# test_function_code --------------------

def test_generate_video():
    """
    Test the generate_video function.
    """
    video_path = generate_video('(redshift style) Spider-Man is water skiing', 8, 512, 512, 50, 7.5)
    assert os.path.exists(video_path), 'Video file does not exist.'
    video_path = generate_video('(redshift style) A cat is dancing', 8, 512, 512, 50, 7.5)
    assert os.path.exists(video_path), 'Video file does not exist.'
    video_path = generate_video('(redshift style) A dog is running', 8, 512, 512, 50, 7.5)
    assert os.path.exists(video_path), 'Video file does not exist.'
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_generate_video())