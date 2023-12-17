# requirements_file --------------------

!pip install -U diffusers transformers controlnet_aux

# function_import --------------------

from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image

# function_code --------------------

def estimate_human_pose(image_path, text_prompt, num_inference_steps=20):
    """
    Estimates the human pose of an actor from an image using a pre-trained ControlNetModel.

    Args:
        image_path (str): The filepath to the image of the actor.
        text_prompt (str): The text description of the desired pose.
        num_inference_steps (int): The number of inference steps for the model. Default is 20.

    Returns:
        Image: The output image with the estimated human pose.

    Raises:
        FileNotFoundError: If the image_path does not exist.
        RuntimeError: If there is an error during model inference.
    """
    try:
        # Load the OpenposeDetector
        openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        
        # Load the image
        image = load_image(image_path)
        image = openpose(image)
        
        # Load the pre-trained ControlNetModel
        controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-openpose', torch_dtype=torch.float16)
        
        # Create the pipeline with the ControlNetModel
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            'runwayml/stable-diffusion-v1-5', controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
        )
        
        # Add additional configurations
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()

        # Estimate the human pose
        output_image = pipe(text_prompt, image, num_inference_steps=num_inference_steps).images[0]
        return output_image
    except FileNotFoundError as e:
        raise FileNotFoundError(f'The image file does not exist: {e}')
    except Exception as e:
        raise RuntimeError(f'Error during model inference: {e}')

# test_function_code --------------------

def test_estimate_human_pose():
    print("Testing started.")
    
    test_cases = [
        ('assets/sample_actor_image1.png', 'actor performing a happy scene', 20),
        ('assets/sample_actor_image2.png', 'actor performing a sad scene', 20),
        ('nonexistent.png', 'image does not exist', 20)
    ]
    
    for i, (image_path, text_prompt, steps) in enumerate(test_cases):
        test_num = i + 1
        try:
            print(f"Testing case [{test_num}/{len(test_cases)}] started.")
            output_image = estimate_human_pose(image_path, text_prompt, steps)
            assert output_image is not None, f"Test case [{test_num}/{len(test_cases)}] failed: Output image is None."
        except (FileNotFoundError, RuntimeError) as e:
            assert isinstance(e, FileNotFoundError), f"Test case [{test_num}/{len(test_cases)}] failed: {e}"
    
    print("Testing finished.")

# call_test_function_line --------------------

test_estimate_human_pose()