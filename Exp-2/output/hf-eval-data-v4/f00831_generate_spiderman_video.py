# requirements_file --------------------

!pip install -U torch, tuneavideo

# function_import --------------------

from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid
import torch

# function_code --------------------

def generate_spiderman_video(prompt, video_length=8, height=512, width=512, num_inference_steps=50, guidance_scale=7.5):
    # Import the required libraries and modules.
    from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
    from tuneavideo.models.unet import UNet3DConditionModel
    from tuneavideo.util import save_videos_grid
    import torch

    # Load the pretrained UNet3DConditionModel and pipeline.
    unet = UNet3DConditionModel.from_pretrained('Tune-A-Video-library/redshift-man-skiing', subfolder='unet', torch_dtype=torch.float16).to('cuda')
    pipe = TuneAVideoPipeline.from_pretrained('nitrosocke/redshift-diffusion', unet=unet, torch_dtype=torch.float16).to('cuda')
    pipe.enable_xformers_memory_efficient_attention()

    # Generate the video using the pipeline.
    video = pipe(prompt, video_length=video_length, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).videos

    # Save the generated video as a file.
    save_videos_grid(video, f'./{prompt}.gif')

    return f'./{prompt}.gif'

# test_function_code --------------------

def test_generate_spiderman_video():
    # Test generating a Spider-Man themed video in redshift style.
    print("Testing generate_spiderman_video()")
    prompt = '(redshift style) Spider-Man is water skiing'
    video_file = generate_spiderman_video(prompt)
    assert os.path.isfile(video_file), f"Test failed: {video_file} does not exist."

    print("Test passed. Generated video file: ", video_file)

# Run the test function
test_generate_spiderman_video()