# function_import --------------------

from diffusers import DDPMPipeline
import os

# function_code --------------------

def generate_insect_image(model_name: str) -> None:
    '''
    Generate an insect image using a pretrained model.

    Args:
        model_name (str): The name of the pretrained model.

    Returns:
        None

    Raises:
        ModuleNotFoundError: If the diffusers package is not installed.
        Exception: If there is an error in generating the image.
    '''
    
    # Load model and create pipeline instance.
    try:
        from diffusers import DiffuserPipeline, get_model
        
        model = get_model(model_name)
        pipeline = DDPMPipeline()
        pipeline.add_model(model=model)
        
    except ModuleNotFoundError as e:
        print("Please install the diffusers package before proceeding.")
    
    # Generate image.
    try:
        pipeline.run_pipeline()
        print('Image saved to "output/insects.png"')
            
    except Exception as e:
        print(f'An error occured when generating the image: {e}')
        
# __main__ -----------------------
    
if __name__ == '__main__':
    '''Generate an insect image using a pretrained model.'''
    
    # Create directory output if it does not exist.
    try:
        os.mkdir('output')
        
    except FileExistsError as e:
        pass
        
    # Generate images.
    models = ['BigGAN-deep-28', 'ProGAN', 'StyleGAN']
    
    for model in models:
        print(f'Generating insect image with {model}...')
        generate_insect_image(model)

# test_function_code --------------------

def test_generate_insect_image():
    '''
    Test the generate_insect_image function.

    Returns:
        str: A message indicating that all tests passed.
    '''
    try:
        generate_insect_image('schdoel/sd-class-AFHQ-32')
        assert os.path.exists('insect_image.png'), 'The image file does not exist.'
        return 'All Tests Passed'
    except Exception as e:
        return str(e)


# call_test_function_code --------------------

if __name__ == '__main__':
    print(test_generate_insect_image())