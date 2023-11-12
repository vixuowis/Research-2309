# function_import --------------------

from pyannote.audio import Pipeline

# function_code --------------------

def speaker_diarization(audio_file: str) -> dict:
    '''
    Function to perform speaker diarization on an audio file using pyannote.audio.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        dict: A dictionary containing speaker diarization results.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        Exception: If any error occurs during the diarization process.
    '''
    try:
        # Load the pretrained model
        pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization@2.1')
        # Perform speaker diarization
        diarization = pipeline(audio_file)
        return diarization
    except FileNotFoundError as fnf_error:
        print(f'Error: {fnf_error}')
        raise
    except Exception as e:
        print(f'Error: {e}')
        raise

# test_function_code --------------------

def test_speaker_diarization():
    '''
    Function to test the speaker_diarization function.
    '''
    # Test with a valid audio file
    try:
        result = speaker_diarization('valid_audio.wav')
        assert isinstance(result, dict), 'Result should be a dictionary.'
    except Exception as e:
        print(f'Error: {e}')
        raise
    # Test with a non-existent audio file
    try:
        result = speaker_diarization('non_existent_audio.wav')
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f'Error: {e}')
        raise
    # Test with an invalid audio file
    try:
        result = speaker_diarization('invalid_audio.wav')
    except Exception as e:
        print(f'Error: {e}')
        raise
    return 'All Tests Passed'

# call_test_function_code --------------------

test_speaker_diarization()