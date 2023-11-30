# function_import --------------------

from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, InterpolationMode, Normalize

# function_code --------------------

def create_artistic_variations(image_path: str, output_path: str) -> None:
    '''
    Create artistic variations of an input image using StableDiffusionImageVariationPipeline.

    Args:
        image_path (str): The path to the input image.
        output_path (str): The path to save the output image.

    Returns:
        None
    '''
    
    transform = Compose([
        lambda img: Image.open(img),
        Resize(256, interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    untransform = lambda img: Image.open(img.permute(1,2,0).byte().cpu())

    image_tensor = transform(image_path)
    pipeline = StableDiffusionImageVariationPipeline(image=image_tensor[None], 
                                                      timesteps=1000, 
                                                      n_images=25)
    
    variations = pipeline.reconstruct()
    for i in range(variations.size(1)):
        untransform(variations[:,i]).save('{}.jpg'.format(output_path))


# test_function_code --------------------

def test_create_artistic_variations():
    '''
    Test the function create_artistic_variations.
    '''
    import os
    import requests
    from PIL import Image
    from io import BytesIO

    # Download a test image
    url = 'https://placekitten.com/200/300'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.save('test.jpg')

    # Apply the function
    create_artistic_variations('test.jpg', 'output.jpg')

    # Check the output
    assert os.path.exists('output.jpg'), 'Output image does not exist.'

    # Clean up
    os.remove('test.jpg')
    os.remove('output.jpg')

    return 'All Tests Passed'


# call_test_function_code --------------------

test_create_artistic_variations()