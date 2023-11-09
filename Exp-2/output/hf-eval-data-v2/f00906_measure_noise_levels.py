# function_import --------------------

from pyannote.audio import Model, Inference

# function_code --------------------

def measure_noise_levels(audio_file_path, access_token):
    """
    This function measures the noise levels in the environment using a pre-trained model from Hugging Face Transformers.
    
    Args:
        audio_file_path (str): The path to the audio file to be analyzed.
        access_token (str): The access token for Hugging Face Transformers.
    
    Returns:
        A list of tuples, each containing the time (in seconds), voice activity detection (VAD) percentage, speech-to-noise ratio (SNR), and C50 room acoustics estimation for each frame in the audio file.
    """
    model = Model.from_pretrained('pyannote/brouhaha', use_auth_token=access_token)
    inference = Inference(model)
    output = inference(audio_file_path)
    results = []
    for frame, (vad, snr, c50) in output:
        t = frame.middle
        results.append((t, 100*vad, snr, c50))
    return results

# test_function_code --------------------

def test_measure_noise_levels():
    """
    This function tests the measure_noise_levels function by providing a sample audio file and checking the output.
    """
    results = measure_noise_levels('sample_audio.wav', 'ACCESS_TOKEN_GOES_HERE')
    assert isinstance(results, list)
    assert all(isinstance(item, tuple) and len(item) == 4 for item in results)

# call_test_function_code --------------------

test_measure_noise_levels()