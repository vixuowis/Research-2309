# requirements_file --------------------

!pip install -U diffusers, Pillow, torch, controlnet_aux

# function_import --------------------

from PIL import Image
import torch
from controlnet_aux import NormalBaeDetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

# function_code --------------------

def generate_book_cover(input_image_path, output_image_path, seed=33):
    # Load the input image depicting love and roses
    image = Image.open(input_image_path)

    # Create a prompt for the generation
    prompt = "A head full of roses"

    # Initialize NormalBaeDetector and process the image
    processor = NormalBaeDetector.from_pretrained('lllyasviel/Annotators')
    control_image = processor(image)

    # Load the ControlNetModel with the given checkpoint
    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_normalbae', torch_dtype=torch.float16)

    # Initialize the StableDiffusionControlNetPipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # Set seed for generation consistency
    generator = torch.manual_seed(seed)

    # Generate the image
    generated_image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]

    # Save the generated image
    generated_image.save(output_image_path)

    return output_image_path

# test_function_code --------------------

def test_generate_book_cover():
    print("Testing generate_book_cover function.")

    output_path = 'test_output.png'

    # Test case: Generate book cover image
    print("Testing book cover generation.")
    generated_path = generate_book_cover('input_image.png', output_path)

    # Validate the output image exists
    assert os.path.exists(output_path), f"Test failed: Generated book cover image not found at {output_path}."

    # Optional: Additional tests can be done such as image size checks, or if in an actual usage scenario,
    # performing visual inspection or more advanced image analysis.

    print("Testing finished successfully.")

# Run the test function
test_generate_book_cover()