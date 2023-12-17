# requirements_file --------------------

!pip install -U pyannote.audio

# function_import --------------------

from pyannote.audio import Pipeline

# function_code --------------------

def diarize_audio(audio_file, access_token):
    """
    Applies speaker diarization to an audio file using a pretrained pyannote.audio model.

    Args:
        audio_file (str): Path to the audio file to be processed.
        access_token (str): Access token for using the pretrained model.

    Returns:
        Diarization: An object containing the diarization results.

    Raises:
        FileNotFoundError: If the audio file is not found.
        ValueError: If the access token is not valid.
    """
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"The audio file {audio_file} was not found.")
    if not access_token:
        raise ValueError("An access token must be provided.")

    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization@2.1', use_auth_token=access_token)
    return pipeline(audio_file)

# test_function_code --------------------

def test_diarize_audio():
    print("Testing started.")
    # Assuming a test audio file and valid access token are available
    test_audio_file = 'test_meeting_audio.wav'
    valid_access_token = 'test_access_token'

    # Test case 1: Audio file exists and token is valid
    print("Testing case [1/3] started.")
    try:
        diarization = diarize_audio(test_audio_file, valid_access_token)
        assert diarization is not None, "Test case [1/3] failed: Diarization object is None"
    except Exception as e:
        assert False, f"Test case [1/3] failed: {e}"

    # Test case 2: Audio file does not exist
    print("Testing case [2/3] started.")
    try:
        diarize_audio('non_existent_audio.wav', valid_access_token)
        assert False, "Test case [2/3] failed: No exception for missing audio file"
    except FileNotFoundError:
        assert True
    except Exception as e:
        assert False, f"Test case [2/3] failed: {e}"

    # Test case 3: Access token is not provided
    print("Testing case [3/3] started.")
    try:
        diarize_audio(test_audio_file, '')
        assert False, "Test case [3/3] failed: No exception for missing access token"
    except ValueError:
        assert True
    except Exception as e:
        assert False, f"Test case [3/3] failed: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_diarize_audio()