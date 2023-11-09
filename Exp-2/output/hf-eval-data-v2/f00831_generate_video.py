# function_import --------------------

from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid
import torch

# function_code --------------------

def generate_video(prompt, video_length=8, height=512, width=512, num_inference_steps=50, guidance_scale=7.5):
    """
    Generate a video based on a textual prompt using a pretrained model.

    Args:
        prompt (str): Textual description of the desired video.
        video_length (int, optional): Length of the generated video. Default is 8.
        height (int, optional): Height of the generated video. Default is 512.
        width (int, optional): Width of the generated video. Default is 512.
        num_inference_steps (int, optional): Number of inference steps. Default is 50.
        guidance_scale (float, optional): Guidance scale. Default is 7.5.

    Returns:
        None. The generated video is saved as a GIF file.
    """
    unet = UNet3DConditionModel.from_pretrained('Tune-A-Video-library/redshift-man-skiing', subfolder='unet', torch_dtype=torch.float16).to('cuda')
    pipe = TuneAVideoPipeline.from_pretrained('nitrosocke/redshift-diffusion', unet=unet, torch_dtype=torch.float16).to('cuda')
    pipe.enable_xformers_memory_efficient_attention()
    video = pipe(prompt, video_length=video_length, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).videos
    save_videos_grid(video, f'./{prompt}.gif')

# test_function_code --------------------

def test_generate_video():
    """
    Test the generate_video function.

    Raises:
        AssertionError: If the function does not work as expected.
    """
    prompt = '(redshift style) Spider-Man is water skiing'
    generate_video(prompt)
    assert os.path.exists(f'./{prompt}.gif'), 'Video file does not exist.'

# call_test_function_code --------------------

test_generate_video()