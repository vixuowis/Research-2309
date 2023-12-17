# requirements_file --------------------

!pip install -U diffusers transformers accelerate controlnet_aux Pillow numpy 

# function_import --------------------

from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image

# function_code --------------------

def estimate_human_pose(image_path, text_prompt, save_path):
    # Load the openpose detector
    openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

    # Load and preprocess the image
    image = load_image(image_path)
    image = openpose(image)

    # Load the controlnet model
    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-openpose', torch_dtype=torch.float16)

    # Setup the pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16
    )

    # Configure the pipeline
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    # Generate the pose image
    result = pipe(text_prompt, image, num_inference_steps=20).images[0]

    # Save the result
    result.save(save_path)
    return save_path

# test_function_code --------------------

def test_estimate_human_pose():
    print("Testing started.")

    # Load a sample data path
    sample_image_path = 'tests/sample_actor_image.png'
    sample_text_prompt = 'actor performing a scene'
    save_path = 'tests/sample_actor_pose_out.png'

    # Test case 1: Check if the function saves an image file
    print("Testing case [1/1] started.")
    result_path = estimate_human_pose(sample_image_path, sample_text_prompt, save_path)
    assert os.path.isfile(result_path), f"Test case [1/1] failed: No image file saved at {result_path}"
    print("Testing finished.")

# Run the test function
if __name__ == '__main__':
    test_estimate_human_pose()