# requirements_file --------------------

!pip install -U pyannote.audio

# function_import --------------------

from pyannote.audio import Pipeline

# function_code --------------------

def perform_speaker_diarization(audio_file_path: str) -> None:
    """Perform speaker diarization on an audio file using pyannote.audio.

    Args:
        audio_file_path (str): The path to the input audio file.

    Returns:
        None: The function writes the diarization result to an RTTM file.

    Raises:
        FileNotFoundError: If the audio file is not found.
        RuntimeError: If the diarization process fails.
    """
    try:
        # Create a pipeline for speaker diarization
        diarization_pipeline = Pipeline.from_pretrained('philschmid/pyannote-speaker-diarization-endpoint')

        # Perform diarization on the specified audio file
        diarization = diarization_pipeline(audio_file_path)

        # Write the diarization result to an RTTM file
        with open("output_audio.rttm", "w") as rttm_file:
            diarization.write_rttm(rttm_file)

    except FileNotFoundError as e:
        raise FileNotFoundError('The specified audio file was not found.') from e
    except Exception as e:
        raise RuntimeError('The diarization process failed.') from e

# test_function_code --------------------

def test_perform_speaker_diarization():
    print("Testing started.")
    # Assume there is an audio file sample.wav in the current directory for testing purposes
    audio_file_path = 'sample.wav'

    # Test case 1: Check if the function executes without throwing an error
    print("Testing case [1/1] started.")
    try:
        perform_speaker_diarization(audio_file_path)
        print("Test case [1/1] passed.")
    except Exception as e:
        print(f"Test case [1/1] failed: {e}")

    print("Testing finished.")

# call_test_function_line --------------------

test_perform_speaker_diarization()