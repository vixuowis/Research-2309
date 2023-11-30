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

    # Load model -----------------------

    diffusion = DiffusionPipeline(model_name)

    # Inference ------------------------
    
    torch_prompt = torch.cat([torch.zeros((1, 50257), dtype=torch.int64)]*num_inference_steps)

    video_tokens = diffusion.generate(prompt="", num_inference_steps=num_inference_steps, prompts_batches=[torch_prompt], use_only_first_prompt=True)['video']

    # Postprocess ----------------------
    
    video = video_tokens[0].float().permute(1, 2, 3, 0).cpu().numpy() / 255.0
    video = (video * 255).astype('uint8')
    video = video[:,:,::-1] # bgr -> rgb

    return export_to_video(video)

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