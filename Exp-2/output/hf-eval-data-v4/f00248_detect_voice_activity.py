# requirements_file --------------------

!pip install -U pyannote.audio==2.1.1

# function_import --------------------

from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model

# function_code --------------------

def detect_voice_activity(audio_file_path, access_token):
    """
    Detects voice activity in a given podcast audio file using a pre-trained model.

    :param audio_file_path: Path to the audio file to be analyzed.
    :param access_token: Token to authenticate and access the pre-trained model.
    :return: An object indicating the segments where voice activity is detected.
    """
    model = Model.from_pretrained('pyannote/segmentation', use_auth_token=access_token)
    pipeline = VoiceActivityDetection(segmentation=model)
    HYPER_PARAMETERS = {
        'onset': 0.5, 'offset': 0.5,
        'min_duration_on': 0.0,
        'min_duration_off': 0.0
    }
    pipeline.instantiate(HYPER_PARAMETERS)
    vad_result = pipeline(audio_file_path)
    return vad_result

# test_function_code --------------------

def test_detect_voice_activity():
    print("Testing detect_voice_activity function.")
    audio_file_path = 'test_audio.wav'  # Replace with a test audio file path
    access_token = 'YOUR_ACCESS_TOKEN'  # Replace with a valid access token

    # Test case: Check if the function returns results without errors
    try:
        vad_result = detect_voice_activity(audio_file_path, access_token)
        assert vad_result is not None, "Function did not return any results."
    except Exception as e:
        assert False, f"Function raised an exception: {str(e)}"

    print("Testing completed successfully.")

# Running the test function
test_detect_voice_activity()