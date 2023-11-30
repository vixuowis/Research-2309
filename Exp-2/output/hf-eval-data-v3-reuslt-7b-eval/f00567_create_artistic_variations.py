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

    # Setup --------------------

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = StableDiffusionImageVariationPipeline(device)
    model = model.to(device)

    transforms = Compose([Resize((model.image_size, model.image_size), interpolation=InterpolationMode.BICUBIC), ToTensor(), Normalize((0.5, 0.5, 0.5), (1, 1, 1)), lambda t: t[:3, ...]])
    
    # Load image and process --------------------

    img_pil = Image.open(image_path)
    img_tensor = transforms(img_pil).unsqueeze(0).to(device)

    output = model.run(img_tensor, 100)
    pil_transforms = Compose([ToTensor()])
    
    # Save image --------------------

    torchvision.utils.save_image((pil_transforms(output[0].cpu())), output_path)

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