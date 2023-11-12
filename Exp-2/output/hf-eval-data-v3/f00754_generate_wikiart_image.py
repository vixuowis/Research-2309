# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_wikiart_image():
    '''
    Generate a new piece of art to add to a digital gallery that resembles WikiArt images.
    
    Returns:
        image: A generated image that resembles images from the WikiArt dataset.
    '''
    pipeline = DDPMPipeline.from_pretrained('johnowhitaker/sd-class-wikiart-from-bedrooms')
    image = pipeline().images[0]
    return image

# test_function_code --------------------

def test_generate_wikiart_image():
    '''
    Test the function generate_wikiart_image.
    '''
    image = generate_wikiart_image()
    assert image is not None, 'The generated image should not be None.'
    assert isinstance(image, type(pipeline().images[0])), 'The generated image should be of the same type as the images generated by the pipeline.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_wikiart_image()