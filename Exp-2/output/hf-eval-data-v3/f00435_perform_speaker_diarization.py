# function_import --------------------

import os
from pyannote.audio import Pipeline

# function_code --------------------

def perform_speaker_diarization(audio_file: str, output_file: str) -> None:
    """
    Perform speaker diarization on an audio file using a pre-trained model from pyannote.audio.

    Args:
        audio_file (str): Path to the audio file.
        output_file (str): Path to the output file where the result will be written in RTTM format.

    Returns:
        None

    Raises:
        FileNotFoundError: If the audio file does not exist.
        Exception: If there is an error in performing speaker diarization.
    """
    try:
        # Check if the audio file exists
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"{audio_file} does not exist.")

        # Instantiate the pre-trained model
        diarization_pipeline = Pipeline.from_pretrained('philschmid/pyannote-speaker-diarization-endpoint')

        # Perform speaker diarization
        diarization = diarization_pipeline(audio_file)

        # Write the result to the output file in RTTM format
        with open(output_file, "w") as rttm:
            diarization.write_rttm(rttm)
    except Exception as e:
        raise Exception(f"Error in performing speaker diarization: {str(e)}")

# test_function_code --------------------

def test_perform_speaker_diarization():
    """
    Test the perform_speaker_diarization function.
    """
    # Test case 1: Valid audio file
    try:
        perform_speaker_diarization('valid_audio_file.wav', 'output_audio.rttm')
    except Exception as e:
        assert False, f"Test case 1 failed: {str(e)}"

    # Test case 2: Non-existent audio file
    try:
        perform_speaker_diarization('non_existent_audio_file.wav', 'output_audio.rttm')
    except FileNotFoundError:
        assert True
    except Exception as e:
        assert False, f"Test case 2 failed: {str(e)}"

    # Test case 3: Invalid audio file format
    try:
        perform_speaker_diarization('invalid_audio_file.txt', 'output_audio.rttm')
    except Exception as e:
        assert str(e) == 'Error in performing speaker diarization: Invalid audio file format.', f"Test case 3 failed: {str(e)}"

    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_perform_speaker_diarization())