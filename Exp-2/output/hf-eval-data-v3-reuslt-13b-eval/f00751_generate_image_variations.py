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

    # Load and prepare an example image
    img = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize((480, 480)), transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
    
    img = transform(img).unsqueeze(dim=0)
    target_image = Image.open("assets/target-images/flower.jpg")
    target_image = transform(target_image).unsqueeze(dim=0)
    
    # Define the model pipeline and generate the variation
    model = StableDiffusionImageVariationPipeline(guidance_scale=guidance_scale, lr=1e-4, num_variations=32)
    model.generate(img, target_image=target_image, output_path=output_path)

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