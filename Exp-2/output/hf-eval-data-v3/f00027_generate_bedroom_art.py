# function_import --------------------

try:
    from diffusers import DDPMPipeline
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install the 'diffusers' library.")

# function_code --------------------

def generate_bedroom_art():
    '''
    Generate a new image based on the online database of bedroom art.
    
    Args:
        None
    
    Returns:
        PIL.Image: An image object that can be displayed or saved.
    
    Raises:
        ModuleNotFoundError: If the 'diffusers' library is not installed.
    '''
    pipeline = DDPMPipeline.from_pretrained('johnowhitaker/sd-class-wikiart-from-bedrooms')
    generated_image = pipeline().images[0]
    return generated_image

# test_function_code --------------------

def test_generate_bedroom_art():
    '''
    Test the generate_bedroom_art function.
    
    Args:
        None
    
    Returns:
        str: 'All Tests Passed' if all assertions pass.
    
    Raises:
        AssertionError: If any of the assertions fail.
    '''
    generated_image = generate_bedroom_art()
    assert isinstance(generated_image, Image.Image), 'The returned object is not an image.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_bedroom_art()