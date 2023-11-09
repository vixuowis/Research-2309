# function_import --------------------

from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model

# function_code --------------------

def detect_voice_activity(audio_file_path, auth_token):
    """
    Detects voice activity in an audio file using a pre-trained model from Hugging Face Transformers.

    Args:
        audio_file_path (str): The path to the audio file.
        auth_token (str): The authentication token for Hugging Face Transformers.

    Returns:
        vad (pyannote.core.SlidingWindowFeature): The voice activity detection result.
    """
    model = Model.from_pretrained('pyannote/segmentation', use_auth_token=auth_token)
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
    """
    Tests the detect_voice_activity function.
    """
    audio_file_path = 'test_audio.wav'  # replace with the path to a test audio file
    auth_token = 'test_auth_token'  # replace with a test authentication token
    vad = detect_voice_activity(audio_file_path, auth_token)
    assert isinstance(vad, pyannote.core.SlidingWindowFeature), 'The return type is incorrect.'

# call_test_function_code --------------------

test_detect_voice_activity()