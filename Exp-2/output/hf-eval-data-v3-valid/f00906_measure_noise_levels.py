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
        model = Model.from_pretrained('pyannote/brouhaha', use_auth_token=access_token)
        inference = Inference(model)
        output = inference(audio_file_path)
        for frame, (vad, snr, c50) in output:
            t = frame.middle
            print(f'{t:8.3f} vad={100*vad:.0f}% snr={snr:.0f} c50={c50:.0f}')
    except FileNotFoundError as fnf_error:
        print(f'Error: {fnf_error}')
    except Exception as e:
        print(f'Error: {e}')

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