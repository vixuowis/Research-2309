# requirements_file --------------------

!pip install -U pyannote.audio

# function_import --------------------

from pyannote.audio import Pipeline

# function_code --------------------

def run_speaker_diarization(audio_file_path: str, access_token: str) -> None:
    """Runs speaker diarization on the provided audio file using pyannote.audio's diarization pipeline.

    Args:
        audio_file_path: The path to the audio file on which to perform diarization.
        access_token: The authorization token for using pre-trained models on pyannote.audio.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError: If the diarization process fails.
    """
    try:
        pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization@2.1', use_auth_token=access_token)
        diarization = pipeline(audio_file_path)
        with open('audio.rttm', 'w') as rttm_file:
            diarization.write_rttm(rttm_file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}") from e
    except Exception as e:
        raise RuntimeError("An error occurred during speaker diarization.") from e


# test_function_code --------------------

import os
from pyannote.audio import Pipeline
def test_run_speaker_diarization():
    print("Testing started.")

    # Prepare a dummy audio file and a dummy token
    dummy_audio_path = 'dummy_audio.wav'
    dummy_token = 'dummy_token'
    try:
        if not os.path.exists(dummy_audio_path):
            raise FileNotFoundError

        # Test case 1
        print("Testing case [1/1] started.")
        run_speaker_diarization(dummy_audio_path, dummy_token)
        assert os.path.isfile('audio.rttm'), f"Test case [1/1] failed: audio.rttm file not created."
        print("Testing finished.")
    except FileNotFoundError:
        print(f"Test case [1/1] failed: Dummy audio file {dummy_audio_path} not found.")
    except Exception as e:
        print(f"Test case [1/1] failed: {str(e)}")

    # Cleanup
    if os.path.isfile('audio.rttm'):
        os.remove('audio.rttm')


# call_test_function_line --------------------

test_run_speaker_diarization()