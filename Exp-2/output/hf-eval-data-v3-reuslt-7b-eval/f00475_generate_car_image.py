# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_car_image(model_id: str = 'google/ddpm-cifar10-32', output_file: str = 'ddpm_generated_image.png'):
    '''
    Generate a car image using the specified pre-trained model.

    Args:
        model_id: The ID of the pre-trained model to use for image generation. Default is 'google/ddpm-cifar10-32'.
        output_file: The name of the file to save the generated image to. Default is 'ddpm_generated_image.png'.

    Returns:
        None. The generated image is saved to a file.

    Raises:
        ModuleNotFoundError: If the diffusers module is not installed.
    '''
    
    try:
        import tensorflow as tf
        
        # Setup TensorFlow to allow for GPU use on MacOS X (Darwin) machines, if possible.
        gpus = tf.config.list_physical_devices('GPU') 
        if gpus: 
            try: 
                # Allow memory growth for each GPU.
                gpu_options = ['allow_growth']
                
                # Set the visible devices to those that can be found on this machine.
                visible_devices = [i for i in range(len(gpus))]
                tf.config.set_visible_devices(gpus, 'GPU')
                logical_gpus = tf.config.list_logical_devices('GPU') 
                
                # Configure TensorFlow to use the GPUs with specified options.
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print('ERROR - Could not allocate GPU.') 
                print(e)
            
        # Create a pipeline for image generation with the specified model ID.
        pipe = DDPMPipeline(model_id=model_id)
        
        # Setup the initial parameters to generate an image from.
        params0, _ = pipe.init_params(shape=[1, 32, 32, 3], device='cpu')  
    
    except ModuleNotFoundError:
        raise ModuleNotFoundError('Could not find required module "diffusers". Please install the package via `pip install diffusers`.')
        
    # Generate an image with the specified parameters.
    img_sampled, params1 = pipe.ddpm(params0=params0)
    
    # Save the generated image to a file.
    tf.io.write_file(output_file, tf.image.encode_png(tf.clip_by_value(img_sampled[0], 0., 1.), compression=-1))

# test_function_code --------------------

def test_generate_car_image():
    '''
    Test the generate_car_image function.

    Returns:
        'All Tests Passed' if all assertions pass.
    '''
    import os

    # Test with default parameters
    generate_car_image()
    assert os.path.exists('ddpm_generated_image.png'), 'Test failed: Default output file not found.'

    # Test with custom parameters
    generate_car_image(model_id='google/ddpm-cifar10-32', output_file='custom_output.png')
    assert os.path.exists('custom_output.png'), 'Test failed: Custom output file not found.'

    return 'All Tests Passed'


# call_test_function_code --------------------

print(test_generate_car_image())