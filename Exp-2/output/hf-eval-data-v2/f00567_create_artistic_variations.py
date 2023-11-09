# function_import --------------------

from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, InterpolationMode, Normalize

# function_code --------------------

def create_artistic_variations(image_path: str, output_path: str = 'result.jpg', guidance_scale: int = 3):
    """
    Create artistic variations of an input image using StableDiffusionImageVariationPipeline.

    Args:
        image_path (str): Path to the input image.
        output_path (str, optional): Path to save the output image. Defaults to 'result.jpg'.
        guidance_scale (int, optional): Guidance scale for the pipeline. Defaults to 3.

    Returns:
        None
    """
    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained('lambdalabs/sd-image-variations-diffusers', revision='v2.0')
    im = Image.open(image_path)
    tform = Compose([
        ToTensor(),
        Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=False),
        Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])
    inp = tform(im).unsqueeze(0)
    out = sd_pipe(inp, guidance_scale=guidance_scale)
    out['images'][0].save(output_path)

# test_function_code --------------------

def test_create_artistic_variations():
    """
    Test the function create_artistic_variations.

    Returns:
        None
    """
    # Use a sample image for testing
    image_path = 'path/to/sample_image.jpg'
    output_path = 'path/to/output_image.jpg'
    create_artistic_variations(image_path, output_path)
    # Check if the output image is created
    assert os.path.exists(output_path)

# call_test_function_code --------------------

test_create_artistic_variations()