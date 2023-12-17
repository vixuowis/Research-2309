# requirements_file --------------------

import subprocess

requirements = ["huggingsound", "torch", "librosa", "datasets", "transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from huggingsound import SpeechRecognitionModel

# function_code --------------------

def transcribe_russian_audio_files(audio_paths):
    """
    Transcribe Russian audio files to text.

    Args:
        audio_paths (list[str]): A list of strings containing the file paths of the audio files to be transcribed.

    Returns:
        list[str]: A list of strings containing the transcriptions of the audio files.

    Raises:
        ValueError: If audio_paths is not a list or is empty.
        FileNotFoundError: If any of the audio files is not found.
    """
    if not isinstance(audio_paths, list) or not audio_paths:
        raise ValueError('The audio_paths argument must be a non-empty list.')
    model = SpeechRecognitionModel('jonatasgrosman/wav2vec2-large-xlsr-53-russian')
    try:
        transcriptions = model.transcribe(audio_paths)
    except FileNotFoundError as e:
        raise FileNotFoundError('One or more audio files were not found: ' + str(e))
    return transcriptions

# test_function_code --------------------

def test_transcribe_russian_audio_files():
    print("Testing started.")
    audio_paths_valid = ['/path/to/valid_file1.wav', '/path/to/valid_file2.mp3']
    audio_paths_empty = []
    audio_paths_invalid = ['/path/to/non_existent_file.wav']

    # Testing case 1: Valid audio files
    print("Testing case [1/3] started.")
    try:
        result = transcribe_russian_audio_files(audio_paths_valid)
        assert isinstance(result, list), "Test case [1/3] failed: Result is not a list."
        assert result, "Test case [1/3] failed: No transcriptions returned."
    except Exception as e:
        assert False, f"Test case [1/3] failed: {e}"

    # Testing case 2: Empty list of audio files
    print("Testing case [2/3] started.")
    try:
        transcribe_russian_audio_files(audio_paths_empty)
        assert False, "Test case [2/3] failed: ValueError not raised on empty list."
    except ValueError:
        pass

    # Testing case 3: Non-existent audio files
    print("Testing case [3/3] started.")
    try:
        transcribe_russian_audio_files(audio_paths_invalid)
        assert False, "Test case [3/3] failed: FileNotFoundError not raised on non-existent file."
    except FileNotFoundError:
        pass
    print("Testing finished.")

# call_test_function_line --------------------

test_transcribe_russian_audio_files()