# function_import --------------------

from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image

# function_code --------------------

def estimate_human_pose(image_path: str, text_prompt: str = 'actor performing a scene', num_inference_steps: int = 20) -> None:
    """
    Estimate the human pose of an actor from an image using ControlNetModel from Hugging Face.

    Args:
        image_path (str): The path to the image of the actor.
        text_prompt (str, optional): The text prompt for the pose estimation. Defaults to 'actor performing a scene'.
        num_inference_steps (int, optional): The number of inference steps. Defaults to 20.

    Returns:
        None. The estimated human pose is saved as an image file 'actor_pose_out.png'.
    """
    openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    image = load_image(image_path)
    image = openpose(image)
    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-openpose', torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    image = pipe(text_prompt, image, num_inference_steps=num_inference_steps).images[0]
    image.save('images/actor_pose_out.png')

# test_function_code --------------------

def test_estimate_human_pose():
    """
    Test the function estimate_human_pose.
    """
    # Test with the default parameters
    estimate_human_pose('actor_image_path.png')
    assert os.path.exists('images/actor_pose_out.png')

    # Test with a different text prompt
    estimate_human_pose('actor_image_path.png', text_prompt='actor dancing')
    assert os.path.exists('images/actor_pose_out.png')

    # Test with a different number of inference steps
    estimate_human_pose('actor_image_path.png', num_inference_steps=30)
    assert os.path.exists('images/actor_pose_out.png')

    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_estimate_human_pose())