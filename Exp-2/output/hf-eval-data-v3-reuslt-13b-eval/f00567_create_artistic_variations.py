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

    # load model and set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sdipvp = StableDiffusionImageVariationPipeline(device=device).to(device)
    
    # load image
    pil_image = Image.open(image_path)
    pil_image = pil_image.convert('RGB')
    original_size = (pil_image.width, pil_image.height)
    
    # preprocess image and create target tensor
    transforms = Compose([
        Resize((512, 512), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])
    input_tensor = transforms(pil_image)[:3].unsqueeze(dim=0).to(device)
    target_tensor = input_tensor.clone()
    
    # run inference and save the output image
    sdipvp(input_tensor, target_tensor)
    pil_output = transforms(sdipvp.prediction[-1].clamp(min=-1., max=1.).detach().cpu())[0] \
                 .permute((1, 2, 0)) * 0.5 + 0.5
    
    # resize and save image (this is required to get a good looking image)
    pil_output = Image.fromarray(np.uint8(pil_output * 256.)).resize(original_size, resample=Image.BICUBIC)
    pil_output.save(output_path)
    
# main --------------------
if __name__ == '__main__':
    # create_artistic_variations('image_1024x768.jpg', 'output_1024x768.png')
    # create_artistic_variations('image_512x38

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