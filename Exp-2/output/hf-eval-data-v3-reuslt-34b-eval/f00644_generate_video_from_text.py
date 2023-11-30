# function_import --------------------

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# function_code --------------------

def generate_video_from_text(prompt: str, model_name: str = 'damo-vilab/text-to-video-ms-1.7b', num_inference_steps: int = 25) -> str:
    '''
    Generate a video from a text description using a pretrained model.

    Args:
        prompt (str): The text description to generate the video from.
        model_name (str, optional): The name of the pretrained model to use. Defaults to 'damo-vilab/text-to-video-ms-1.7b'.
        num_inference_steps (int, optional): The number of inference steps to perform. Defaults to 25.

    Returns:
        str: The path to the generated video.
    '''
    
    # Initialize model and pipeline
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    schedule = DPMSolverMultistepScheduler(diffusion_steps = num_inference_steps)
    model = DiffusionPipeline(schedule, device)
    
    # Generate video
    output_video_path = f"{model.generate_token()}.mp4"
    model.load_and_run(output_file=output_video_path, prompt=[prompt], model_name=model_name)
    
    return output_video_path

# test_function_code --------------------

def test_generate_video_from_text():
    '''
    Test the generate_video_from_text function.
    '''
    prompts = ['cats playing with laser pointer', 'Spiderman is surfing', 'A dog chasing its tail']
    for prompt in prompts:
        video_path = generate_video_from_text(prompt)
        assert isinstance(video_path, str)
        assert video_path.endswith('.mp4')

    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_video_from_text()