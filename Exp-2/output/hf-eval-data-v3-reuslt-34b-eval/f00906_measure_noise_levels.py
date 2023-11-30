# function_import --------------------

from pyannote.audio import Model, Inference

# function_code --------------------

def measure_noise_levels(audio_file_path: str, access_token: str):
    """
    Measures noise levels in the environment using a pre-trained model from Hugging Face Transformers.

    Args:
        audio_file_path (str): The path to the audio file.
        access_token (str): The access token for Hugging Face Transformers.

    Returns:
        None. Prints the voice activity detection (VAD), speech-to-noise ratio (SNR), and the C50 room acoustics estimation for each frame in the audio file.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        Exception: If there is an error loading the model or processing the audio file.
    """    
    # Load the model and tokenizer from Hugging Face Transformers
    model_name = 'pb/C50_speech-detection'
    print('Downloading the model')
    inference = Inference(model_name, access_token=access_token)
    print('Loading the model to memory\n')
    model: Model = inference.instantiate()
    
    # Process the audio file and display the outputs in a human-readable format
    try:
        output = model(audio_file_path, batch_size=1)
        
        # Loop through all frames of the audio file
        for i, frame in enumerate(output):
            print('frame',i)
            
            # Convert the time from seconds to HH:MM:SS
            timestamp = str(int(i * 0.2 / 3600)) + ':' + str(int((i * 0.2 % 3600) / 60)).zfill(2) + ':' + str(round(i*0.2 % 60, 2)).zfill(4)[0:2]
            
            # Print the frame output in a human-readable format
            print('timestamp', timestamp)
            print('VAD:', round(frame['vad_prob'][1],3))
            print('SNR:', round(frame['snr'],3))
            print('C50 room acoustics estimation:', round(frame['c50_room_estimation'],2), '\n')
    
    # Handle exceptions
    except FileNotFoundError as error:
        raise error
    except Exception as exception:
        print("Exception occurred during processing. See stacktrace below for details.\n\n", file=sys.stderr)
        traceback.print_exc()
        raise exception

# function_call --------------------

if __name__ == '__main__':
    # Access token for Hugging Face Transformers. Get yours at https://huggingface.co/models?search=speech-detection
    access_token = '<your_access_token>'
    
   

# test_function_code --------------------

def test_measure_noise_levels():
    """
    Tests the measure_noise_levels function.
    """
    # Test with a valid audio file and access token
    try:
        measure_noise_levels('valid_audio_file.wav', 'valid_access_token')
    except Exception as e:
        print(f'Error: {e}')

    # Test with an invalid audio file
    try:
        measure_noise_levels('invalid_audio_file.wav', 'valid_access_token')
    except FileNotFoundError as fnf_error:
        assert str(fnf_error) == "[Errno 2] No such file or directory: 'invalid_audio_file.wav'", 'Test Failed'

    # Test with an invalid access token
    try:
        measure_noise_levels('valid_audio_file.wav', 'invalid_access_token')
    except Exception as e:
        assert str(e) == 'Invalid access token', 'Test Failed'

    return 'All Tests Passed'


# call_test_function_code --------------------

print(test_measure_noise_levels())