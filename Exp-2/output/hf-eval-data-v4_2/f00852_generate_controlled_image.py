# requirements_file --------------------

!pip install -U diffusers transformers accelerate controlnet_aux

# function_import --------------------

import torch
from controlnet_aux import OpenposeDetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image

# function_code --------------------

def generate_controlled_image(text_prompt, input_image_path, output_image_path):
    """
    Generate an image based on a textual description and an input image using a pretrained ControlNet model.

    Args:
        text_prompt (str): The textual description of the scene to generate.
        input_image_path (str): Path to input image for pose detection.
        output_image_path (str): Path to save the generated image.

    Returns:
        str: Path to the generated image.

    Raises:
        FileNotFoundError: If input_image_path does not exist.
        RuntimeError: If there is an error during image generation.
    """
    # Load the pretrained ControlNet model
    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_openpose', torch_dtype=torch.float16)
    
    # Create an OpenPose detector
    openpose_detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    
    # Load and process the input image
    input_image = Image.open(input_image_path)
    control_image = openpose_detector(input_image, hand_and_face=True)
    
    # Create the image generation pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    
    # Set the random seed
    generator = torch.manual_seed(0)
    
    # Generate the image
    output_image = pipe(text_prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    
    # Save the generated image
    output_image.save(output_image_path)
    
    return output_image_path

# test_function_code --------------------

def test_generate_controlled_image():
    print("Testing started.")

    # Define sample data
    text_prompt = 'a person doing yoga in a park'
    input_image_path = 'sample_input.jpg'
    output_image_path = 'generated_image.jpg'

    # Test case 1: Check if function raises FileNotFoundError for non-existing input image
    print("Testing case [1/3] started.")
    with pytest.raises(FileNotFoundError):
        generate_controlled_image(text_prompt, 'non_existing.jpg', output_image_path)

    # Prepare sample input image for other test cases
    create_sample_input_image(input_image_path)

    # Test case 2: Check if function returns proper output image path
    print("Testing case [2/3] started.")
    assert generate_controlled_image(text_prompt, input_image_path, output_image_path) == output_image_path

    # Test case 3: Check if the output image file is created
    print("Testing case [3/3] started.")
    assert os.path.isfile(output_image_path), 'Output image file was not created.'
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_controlled_image()