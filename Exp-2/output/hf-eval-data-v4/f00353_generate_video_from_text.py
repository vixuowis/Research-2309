# requirements_file --------------------

!pip install -U git+https://github.com/huggingface/diffusers transformers accelerate

# function_import --------------------

from diffusers import DiffusionPipeline
from diffusers.schedulers_async import DPMSolverMultistepScheduler
import torch

# function_code --------------------

def generate_video_from_text(prompt, model_name='damo-vilab/text-to-video-ms-1.7b', num_inference_steps=25):
    """
    Generate a video from a textual description using a pre-trained text-to-video diffusion model.

    Args:
    - prompt (str): The textual description to generate the video from.
    - model_name (str, optional): The name of the pre-trained model. Default is 'damo-vilab/text-to-video-ms-1.7b'.
    - num_inference_steps (int, optional): The number of diffusion steps to use. Default is 25.

    Returns:
    - video_frames (torch.Tensor): The tensor containing the generated video frames.
    """
    # Initialize the pipeline with the specified model
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, variant='fp16')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # Generate video frames from the textual description
    video_frames = pipe(prompt, num_inference_steps=num_inference_steps).frames
    
    return video_frames

# test_function_code --------------------

def test_generate_video_from_text():
    print("Testing started.")
    # No dataset required. A textual prompt is provided directly.

    # Testing case 1: Check if function returns a tensor
    print("Testing case [1/3] started.")
    prompt = "a person walking along a beach"
    video_frames = generate_video_from_text(prompt)
    assert isinstance(video_frames, torch.Tensor), f"Test case [1/3] failed: Function did not return a tensor."

    # Testing case 2: Check for non-empty output
    print("Testing case [2/3] started.")
    assert video_frames.nelement() > 0, f"Test case [2/3] failed: Generated video frames tensor is empty."

    # Testing case 3: Check for the correct number of frames (This will depend on the model's output, assuming 25 for this example)
    print("Testing case [3/3] started.")
    assert video_frames.shape[0] == 25, f"Test case [3/3] failed: The number of generated frames does not match the expected output."
    print("Testing finished.")

# Run the test function
test_generate_video_from_text()