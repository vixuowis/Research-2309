# requirements_file --------------------

!pip install -U diffusers transformers accelerate

# function_import --------------------

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# function_code --------------------

def generate_video_from_text(prompt, num_inference_steps=25):
    """
    Generate a video from the given text prompt using a pre-trained text-to-video diffusion model.

    :param prompt: A string containing the text prompt to describe the video content.
    :param num_inference_steps: An integer indicating the number of inference steps for video generation.
    :return: Path to the generated video file.
    """
    # Load the pre-trained model from Hugging Face
    pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b-legacy', torch_dtype=torch.float16)

    # Configure the scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Enable CPU offloading
    pipe.enable_model_cpu_offload()

    # Generate video frames
    video_frames = pipe(prompt, num_inference_steps=num_inference_steps).frames

    # Save and return the path to the video
    video_path = export_to_video(video_frames)
    return video_path

# test_function_code --------------------

def test_generate_video_from_text():
    print("Testing generate_video_from_text function.")

    # Test case: short story prompt
    prompt = "A peaceful sunrise over the mountains"
    print("Testing with a short story prompt.")
    video_path = generate_video_from_text(prompt)
    assert video_path is not None, "Failed to generate video from a short story prompt."

    # Test case: description of a scene
    prompt = "A dog jumping over a white fence on a sunny day"
    print("Testing with a description of a scene.")
    video_path = generate_video_from_text(prompt)
    assert video_path is not None, "Failed to generate video from a scene description."

    print("All tests passed!")

# Run the test function
test_generate_video_from_text()