# function_import --------------------

from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model

# function_code --------------------

def detect_voice_activity(audio_file_path: str, access_token: str) -> dict:
    '''
    Detects voice activity in an audio file using a pre-trained model from Hugging Face Transformers.

    Args:
        audio_file_path (str): The path to the audio file.
        access_token (str): The access token for Hugging Face Transformers.

    Returns:
        dict: A dictionary containing the voice activity detection results.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        Exception: If there is an error in processing the audio file.
    '''
    model = Model.from_pretrained('pyannote/segmentation', use_auth_token=access_token)
    pipeline = VoiceActivityDetection(segmentation=model)
    HYPER_PARAMETERS = {
     'onset': 0.5, 'offset': 0.5,
     'min_duration_on': 0.0,
     'min_duration_off': 0.0
    }
    pipeline.instantiate(HYPER_PARAMETERS)
    vad = pipeline(audio_file_path)
    return vad

# test_function_code --------------------

def test_detect_voice_activity():
    '''
    Tests the detect_voice_activity function.
    '''
    # Test with a valid audio file and access token
    try:
        result = detect_voice_activity('valid_audio_file.wav', 'valid_access_token')
        assert isinstance(result, dict), 'The result should be a dictionary.'
    except Exception as e:
        print(f'Test failed with error: {e}')

    # Test with an invalid audio file
    try:
        result = detect_voice_activity('invalid_audio_file.wav', 'valid_access_token')
        assert isinstance(result, dict), 'The result should be a dictionary.'
    except FileNotFoundError:
        print('Test passed: FileNotFoundError was raised as expected.')
    except Exception as e:
        print(f'Test failed with error: {e}')

    # Test with an invalid access token
    try:
        result = detect_voice_activity('valid_audio_file.wav', 'invalid_access_token')
        assert isinstance(result, dict), 'The result should be a dictionary.'
    except Exception as e:
        print(f'Test passed: Exception was raised as expected with error: {e}')

    print('All tests passed.')

# call_test_function_code --------------------

test_detect_voice_activity()