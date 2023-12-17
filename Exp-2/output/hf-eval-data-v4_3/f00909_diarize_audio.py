# requirements_file --------------------

import subprocess

requirements = ["pyannote.audio"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from pyannote.audio import Pipeline

# function_code --------------------

def diarize_audio(file_path: str, access_token: str, output_path: str = 'audio_file.rttm') -> None:
    """
    Applies speaker diarization to an audio file using pyannote.audio.

    Args:
        file_path (str): The path to the audio file that needs diarization.
        access_token (str): The access token for using the pyannote pretrained model.
        output_path (str): The path where the RTTM output will be saved. Defaults to 'audio_file.rttm'.

    Returns:
        None

    Raises:
        FileNotFoundError: If the input audio file does not exist.
        Exception: If there is an error in the diarization process.
    """
    # Check if the input audio file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'The audio file {file_path} does not exist.')

    # Load the pretrained diarization pipeline
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization@2.1', use_auth_token=access_token)

    # Apply the diarization pipeline to the audio file
    try:
        diarization = pipeline(file_path)

        # Save the diarization output
        with open(output_path, 'w') as rttm_file:
            diarization.write_rttm(rttm_file)
    except Exception as e:
        raise e

# test_function_code --------------------

import os

def test_diarize_audio():
    # Setup: Ensure a test audio file exists
    test_audio_path = 'test_audio.wav'
    test_output_path = 'test_output.rttm'
    test_access_token = 'test_token'

    if not os.path.exists(test_audio_path):
        with open(test_audio_path, 'w') as f:  # Creating a dummy file for test purposes
            f.write('Dummy audio content')

    print('Testing started.')
    try:
        # Test Case 1: Valid audio file path
        print('Testing case [1/3] started.')
        diarize_audio(test_audio_path, test_access_token, test_output_path)
        assert os.path.exists(test_output_path), f'Test case [1/3] failed: RTTM output file not created.'

        # Test Case 2: Invalid audio file path
        print('Testing case [2/3] started.')
        try:
            diarize_audio('nonexistent_audio.wav', test_access_token)
            assert False, 'Test case [2/3] failed: FileNotFoundError not raised for non-existent audio file.'
        except FileNotFoundError:
            pass  # Expected exception

        # Cleanup the created test files
        if os.path.exists(test_audio_path):
            os.remove(test_audio_path)
        if os.path.exists(test_output_path):
            os.remove(test_output_path)
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
    finally:
        print('Testing finished.')

# call_test_function_line --------------------

test_diarize_audio()