# requirements_file --------------------

!pip install -U torch, diffusers, transformers, accelerate

# function_import --------------------

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# function_code --------------------

def generate_video_from_text(description):
    # Load the text-to-video synthesis model
    pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b', torch_dtype=torch.float16, variant='fp16')
    # Configure the scheduler for the model
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # Enable CPU offload to save VRAM
    pipe.enable_model_cpu_offload()
    # Generate video frames from the provided text description
    video_frames = pipe(description, num_inference_steps=25).frames
    # Export the frames to a video file
    video_path = export_to_video(video_frames)
    # Return the path to the generated video file
    return video_path

# test_function_code --------------------

def test_generate_video_from_text():
    print('Testing function generate_video_from_text')
    # Use a sample text description
    sample_description = 'A dog chasing its tail in a park'
    # Generate the video
    video_path = generate_video_from_text(sample_description)
    # Check if the video file was created
    assert os.path.exists(video_path), f'Test failed: The video at {video_path} was not created'
    print('Test passed: Video generated successfully')