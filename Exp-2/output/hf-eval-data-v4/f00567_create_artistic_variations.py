# requirements_file --------------------

!pip install -U diffusers>=0.8.0 torchvision pillow

# function_import --------------------

from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, InterpolationMode, Normalize
import os

# function_code --------------------

def create_artistic_variations(input_image_path, output_image_path):
    """
    Given an input image path, this function generates artistic variations
    of the input image and saves the result to the specified output image path.
    
    Args:
    - input_image_path (str): The file path of the input image.
    - output_image_path (str): The file path where the output image will be saved.

    Returns:
    None. The resulting image is saved to the output_image_path.
    """
    
    # Initialize the pipeline for image variation
    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        'lambdalabs/sd-image-variations-diffusers', revision='v2.0'
    )

    # Load and preprocess the input image
    im = Image.open(input_image_path)
    tform = Compose([
        ToTensor(),
        Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=False),
        Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])
    inp = tform(im).unsqueeze(0)
    
    # Generate artistic variations of the input image
    out = sd_pipe(inp, guidance_scale=3)
    
    # Save the output image
    out['images'][0].save(output_image_path)

# test_function_code --------------------

def test_create_artistic_variations():
    print("Testing started.")
    
    input_image_path = 'path/to/test/input/image.jpg'  # Path to a test input image
    output_image_path = 'path/to/test/output/result.jpg'  # Path to save the output image

    # Test case 1: Check if the output file is created
    print("Testing case [1/1] started.")
    create_artistic_variations(input_image_path, output_image_path)
    assert os.path.exists(output_image_path), f"Test case [1/1] failed: Output file was not created."
    
    print("Testing finished.")