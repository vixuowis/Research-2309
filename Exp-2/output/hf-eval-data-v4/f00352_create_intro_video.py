# requirements_file --------------------

!pip install -U diffusers transformers accelerate

# function_import --------------------

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# function_code --------------------

def create_intro_video(prompt, model_name='damo-vilab/text-to-video-ms-1.7b', num_inference_steps=25, output_path='intro_video.mp4'):
    # Load the pre-trained model with the specified parameters
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, variant='fp16')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # Generate video frames from the prompt string
    video_frames = pipe(prompt, num_inference_steps=num_inference_steps).frames

    # Export the frames to a video file
    export_to_video(video_frames, output_path)
    return output_path

# test_function_code --------------------

def test_create_intro_video():
    print("Testing create_intro_video function.")

    # Define the prompt for the intro video
    prompt = "Chef John's Culinary Adventures"

    # Generate the intro video
    video_path = create_intro_video(prompt)

    # Test if the video file has been created successfully
    print("Testing video file creation.")
    assert os.path.exists(video_path), f"Video file was not created: {video_path}"

    print("Testing create_intro_video function completed successfully.")

# Run the test function
if __name__ == '__main__':
    test_create_intro_video()