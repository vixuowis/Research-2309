# function_import --------------------

import os
import subprocess

# function_code --------------------

def text_to_speech(text: str, model: str = 'mio/amadeus'):
    """
    Convert the input text to speech using the specified model.

    Args:
        text (str): The input text to be converted to speech.
        model (str): The model to be used for text-to-speech conversion. Default is 'mio/amadeus'.

    Returns:
        None
    """
    # Navigate to the ESPnet directory
    os.chdir('espnet')

    # Checkout the specified commit
    subprocess.run(['git', 'checkout', 'd5b5ec7b2e77bd3e10707141818b7e6c57ac6b3f'], check=True)

    # Install the required dependencies
    subprocess.run(['pip', 'install', '-e', '.'], check=True)

    # Navigate to the 'amadeus' recipe directory
    os.chdir('egs2/amadeus/tts1')

    # Download the specified model
    subprocess.run(['./run.sh', '--skip_data_prep', 'false', '--skip_train', 'true', '--download_model', model], check=True)

    # Convert the input text to speech
    subprocess.run(['echo', text, '|', './run.sh', '--stage', '3', '--stop_stage', '3'], check=True)

# test_function_code --------------------

def test_text_to_speech():
    """
    Test the text_to_speech function.
    """
    try:
        # Test with default model
        text_to_speech('Hello, world!')

        # Test with a different model
        text_to_speech('Hello, world!', 'mio/amadeus')

        print('All Tests Passed')
    except Exception as e:
        print(f'Test failed: {e}')

# call_test_function_code --------------------

test_text_to_speech()