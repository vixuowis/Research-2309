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
    try:
        model = Model(
            source="huggingface", 
            checkpoint="julien-c/noise-filter-with-speechbrain"
        )
        noise_estimator = Inference(model, chunk_len=1.6, shift=0.8)
        
    except Exception as exception:
        raise FileNotFoundError("There was an error loading the pre-trained model.") from exception
    
    try:
        with open(audio_file_path, "rb") as file: 
            audio = file.read() 
            
    except Exception as exception:
        raise FileNotFoundError("There was an error opening the audio file.") from exception
        
    try:
        for vad, noise, speech, c50 in noise_estimator(audio, sample_rate=16000, batch_size=32): # sample rate of 16 kHz is recommended for this model. The batch size of 32 may also be helpful.
            print("VAD: {}\tSNR: {:.1f} dB\tc50: {}".format(vad, noise, c50)) # Print the voice activity detection (VAD), speech-to-noise ratio (SNR), and the C50 room acoustics estimation for each frame in the audio file.
        
    except Exception as exception:
        raise Exception("There was an error processing the audio file.") from exception

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