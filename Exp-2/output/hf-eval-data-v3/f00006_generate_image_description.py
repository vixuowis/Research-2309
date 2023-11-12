# function_import --------------------

from transformers import GenerativeImage2TextModel

# function_code --------------------

def generate_image_description(image_path):
    '''
    Generate a product description based on the input image.
    
    Args:
        image_path (str): The path to the image file.
    
    Returns:
        str: The generated product description.
    
    Raises:
        Exception: If the image file cannot be loaded.
    '''
    # Import the required class GenerativeImage2TextModel from the transformers library
    from transformers import GenerativeImage2TextModel
    
    # Load the model using the from_pretrained method and specify the model as 'microsoft/git-large-coco'
    git_model = GenerativeImage2TextModel.from_pretrained('microsoft/git-large-coco')
    
    # Provide the model with the image of the product as input
    product_description = git_model.generate_image_description(image_path)
    
    return product_description

# test_function_code --------------------

def test_generate_image_description():
    '''
    Test the function generate_image_description.
    '''
    # Test case 1: A normal image
    image_path = 'https://placekitten.com/200/300'
    description = generate_image_description(image_path)
    assert isinstance(description, str), 'The output should be a string.'
    
    # Test case 2: An empty image
    image_path = 'https://placekitten.com/g/0/0'
    description = generate_image_description(image_path)
    assert isinstance(description, str), 'The output should be a string.'
    
    # Test case 3: A large image
    image_path = 'https://placekitten.com/1000/1000'
    description = generate_image_description(image_path)
    assert isinstance(description, str), 'The output should be a string.'
    
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_generate_image_description())