# function_import --------------------

import torch
from controlnet_aux import OpenposeDetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

# function_code --------------------

def generate_image_from_text(text_prompt, input_image):
    '''
    Generate an image based on a textual description and an input image.
    
    Args:
        text_prompt (str): The textual description of the scene.
        input_image (PIL.Image): The input image to detect the positions and poses of the objects.
    
    Returns:
        PIL.Image: The generated image.
    '''
    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_openpose', torch_dtype=torch.float16)
    openpose_detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    control_image = openpose_detector(input_image, hand_and_face=True)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    generator = torch.manual_seed(0)
    output_image = pipe(text_prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    return output_image

# test_function_code --------------------

def test_generate_image_from_text():
    '''
    Test the function generate_image_from_text.
    
    This function does not return anything but raises an error if the function
    generate_image_from_text is incorrect.
    '''
    text_prompt = 'chef in the kitchen'
    input_image = torch.rand(3, 256, 256)  # A random image.
    output_image = generate_image_from_text(text_prompt, input_image)
    assert output_image.size == (3, 256, 256), 'The size of the output image is incorrect.'

# call_test_function_code --------------------

test_generate_image_from_text()