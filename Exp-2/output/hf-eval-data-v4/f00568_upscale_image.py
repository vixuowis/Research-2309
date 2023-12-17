# requirements_file --------------------

!pip install -U transformers Pillow

# function_import --------------------

from transformers import Swin2SRForImageSuperResolution
from PIL import Image

# function_code --------------------

def upscale_image(image_path, output_path):
    """
    Upscale the image at the given path by a factor of 2 using the Swin2SR model.
    Args:
        image_path (str): Path to the input image.
        output_path (str): Path where the upscaled image will be saved.
    Returns:
        None: The upscaled image is saved at the output path.
    """
    image = Image.open(image_path)
    model = Swin2SRForImageSuperResolution.from_pretrained('caidas/swin2sr-classical-sr-x2-64')
    upscaled_image = model(image)
    upscaled_image.save(output_path)

# test_function_code --------------------

def test_upscale_image():
    print("Testing upscale_image function.")
    test_image_path = 'test_image.jpg'
    output_image_path = 'test_output.jpg'

    # Assuming test_image.jpg is a valid image file in the current directory
    upscale_image(test_image_path, output_image_path)

    # Check if the file exists and is not empty
    assert os.path.isfile(output_image_path) and os.path.getsize(output_image_path) > 0, f"Upscaling failed, output image not saved properly."

    print("Test passed: Upscaled image saved successfully.")

# Run the test function
test_upscale_image()