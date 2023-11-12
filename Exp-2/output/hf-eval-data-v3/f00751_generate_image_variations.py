# function_import --------------------

from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision import transforms

# function_code --------------------

def generate_image_variations(image_path: str, output_path: str, guidance_scale: int = 3):
    """
    Generate variations of a given image using a pre-trained model.

    Args:
        image_path (str): Path to the original image.
        output_path (str): Path to save the generated image variations.
        guidance_scale (int, optional): Control the number and style of variations. Defaults to 3.

    Returns:
        None
    """
    image = Image.open(image_path)
    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained('lambdalabs/sd-image-variations-diffusers', revision='v2.0')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    ])
    inp = transform(image).unsqueeze(0)
    output = sd_pipe(inp, guidance_scale=guidance_scale)
    output['images'][0].save(output_path)

# test_function_code --------------------

def test_generate_image_variations():
    """
    Test the function generate_image_variations.
    """
    import os
    import requests
    from PIL import Image
    from io import BytesIO

    # Download a test image
    url = 'https://placekitten.com/200/300'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.save('test.jpg')

    # Generate image variations
    generate_image_variations('test.jpg', 'result.jpg')

    # Check if the result image exists
    assert os.path.exists('result.jpg'), 'Result image does not exist.'

    # Clean up
    os.remove('test.jpg')
    os.remove('result.jpg')

    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_image_variations()