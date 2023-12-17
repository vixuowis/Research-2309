# requirements_file --------------------

!pip install -U diffusers torchvision Pillow

# function_import --------------------

from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, InterpolationMode, Normalize

# function_code --------------------

def generate_artistic_variations(input_image_path, output_image_path):
    """
    Apply a pre-trained Stable Diffusion model to generate artistic variations of an input image.

    Args:
        input_image_path (str): The file path of the input image.
        output_image_path (str): The file path where the output image will be saved.

    Returns:
        None

    Raises:
        FileNotFoundError: If the input_image_path does not correspond to a file.
        IOError: If there is an issue saving the output file.
    """
    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        'lambdalabs/sd-image-variations-diffusers', revision='v2.0'
    )
    im = Image.open(input_image_path)
    tform = Compose([
        ToTensor(),
        Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=False),
        Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])
    inp = tform(im).unsqueeze(0)
    out = sd_pipe(inp, guidance_scale=3)
    out['images'][0].save(output_image_path)

# test_function_code --------------------

def test_generate_artistic_variations():
    print("Testing started.")

    input_path = 'test_input.jpg'
    output_path = 'test_output.jpg'

    print("Testing case [1/1] started.")
    try:
        generate_artistic_variations(input_path, output_path)
        # Ensure the output file is created
        assert os.path.exists(output_path), f"Test case [1/1] failed: Output file {output_path} was not created."
    except Exception as e:
        assert False, f"Test case [1/1] failed: {e}" 
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_artistic_variations()