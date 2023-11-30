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
    # load data --------------------
    image = Image.open(image_path)
    image_tensor = transforms.ToTensor()(image).unsqueeze_(0)
    guidance = torch.ones((1, 16 * guidance_scale ** 2)) / (guidance_scale ** 2)  # [B, N]
    # generate image --------------------
    image_variations = StableDiffusionImageVariationPipeline(image_tensor, guidance)

    # save data --------------------
    for i in range(len(image_variations)):
        Image.fromarray((image_variations[i].detach().cpu()[0] * 256).numpy().astype('uint8')).save(f"{output_path}/{i}.jpg")
    print("Generated image variations successfully!")
    

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