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
        input_image (str): The path to the input image file.

    Returns:
        output_image (PIL.Image.Image): The generated image.
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
    '''
    text_prompt = 'A cat sitting on a sofa'
    input_image = 'https://placekitten.com/200/300'
    output_image = generate_image_from_text(text_prompt, input_image)
    assert isinstance(output_image, torch.Tensor), 'The output should be a torch.Tensor'
    assert output_image.shape == (3, 256, 256), 'The shape of the output image should be (3, 256, 256)'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_image_from_text()