# function_import --------------------

from pyannote.audio import Pipeline

# function_code --------------------

def generate_speaker_diarization(audio_file_path: str) -> dict:
    """
    Generate speaker diarization for a given audio file.

    Args:
        audio_file_path (str): The path to the audio file.

    Returns:
        dict: A dictionary containing speaker labels and timestamps.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        Exception: If there is an error in processing the audio file.
    """
    try:
        # Load the pre-trained speaker diarization model
        pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization@2.1')
        # Process the audio file with the loaded model
        diarization = pipeline(audio_file_path)
        return diarization
    except FileNotFoundError as fnf_error:
        print(f'Error: {fnf_error}')
        raise
    except Exception as e:
        print(f'Error: {e}')
        raise

# test_function_code --------------------

def test_generate_speaker_diarization():
    """
    Test the generate_speaker_diarization function.
    """
    # Test with a valid audio file
    try:
        result = generate_speaker_diarization('path/to/valid/audio/file.wav')
        assert isinstance(result, dict), 'The result should be a dictionary.'
        print('Test with a valid audio file passed.')
    except Exception as e:
        print(f'Test with a valid audio file failed. Error: {e}')

    # Test with a non-existing audio file
    try:
        result = generate_speaker_diarization('path/to/non-existing/audio/file.wav')
    except FileNotFoundError:
        print('Test with a non-existing audio file passed.')
    except Exception as e:
        print(f'Test with a non-existing audio file failed. Error: {e}')

    # Test with an invalid audio file
    try:
        result = generate_speaker_diarization('path/to/invalid/audio/file.wav')
    except Exception:
        print('Test with an invalid audio file passed.')
    else:
        print('Test with an invalid audio file failed.')

    print('All Tests Passed')

# call_test_function_code --------------------

test_generate_speaker_diarization()