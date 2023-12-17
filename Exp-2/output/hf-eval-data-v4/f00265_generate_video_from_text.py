# requirements_file --------------------

!pip install -U diffusers torch

# function_import --------------------

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# function_code --------------------

def generate_video_from_text(prompt):
    # Load the pretrained text-to-video model
    pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b', torch_dtype=torch.float16, variant='fp16')
    # Configure the scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # Enable CPU offload for the model
    pipe.enable_model_cpu_offload()
    # Generate video frames from the input text prompt
    video_frames = pipe(prompt, num_inference_steps=25).frames
    # Export video frames to a video file
    video_path = export_to_video(video_frames, 'output_video.mp4')
    # Return the path of the generated video
    return video_path

# test_function_code --------------------

def test_generate_video_from_text():
    print("Testing started.")
    prompt = "A couple sitting in a cafe and laughing while using our product"
    video_path = generate_video_from_text(prompt)
    # Check if the video file was created and exists
    assert os.path.exists(video_path), f"Test failed: Video not created at {video_path}"
    print("Testing finished.")

test_generate_video_from_text()