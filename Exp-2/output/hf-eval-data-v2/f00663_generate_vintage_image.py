# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_vintage_image(model_name: str = 'pravsels/ddpm-ffhq-vintage-finetuned-vintage-3epochs') -> None:
    """
    Generates a vintage image using a pre-trained model from Hugging Face Transformers.

    Args:
        model_name (str): The name of the pre-trained model. Default is 'pravsels/ddpm-ffhq-vintage-finetuned-vintage-3epochs'.

    Returns:
        None. The function saves the generated image to the current directory.
    """
    # Initialize the pipeline with the pre-trained model
    pipeline = DDPMPipeline.from_pretrained(model_name)
    
    # Generate the image
    vintage_image = pipeline().images[0]
    
    # Save the image to the current directory
    vintage_image.save('vintage_magazine_cover.png')

# test_function_code --------------------

def test_generate_vintage_image():
    """
    Tests the function generate_vintage_image.
    """
    # Call the function
    generate_vintage_image()
    
    # Check if the image file was created
    assert os.path.isfile('vintage_magazine_cover.png')

# call_test_function_code --------------------

test_generate_vintage_image()