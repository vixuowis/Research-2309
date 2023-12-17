# requirements_file --------------------

!pip install -U huggingsound torch librosa datasets transformers

# function_import --------------------

from huggingsound import SpeechRecognitionModel

# function_code --------------------

def transcribe_audio(audio_paths):
    """
    Transcribes a list of audio file paths to text using a pre-trained speech recognition model.

    Args:
        audio_paths (list of str): A list of strings containing the file paths to audio files.

    Returns:
        list of str: A list of transcribed texts from the provided audio files.

    Raises:
        ValueError: If the list of audio paths is empty.
        FileNotFoundError: If any audio file is not found at the specified path(s).

    """
    if not audio_paths:
        raise ValueError('The list of audio paths should not be empty.')
    
    try:
        model = SpeechRecognitionModel('jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn')
        transcriptions = model.transcribe(audio_paths)
        return transcriptions
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Audio file not found: {e.filename}')

# test_function_code --------------------

from datasets import load_dataset

def test_transcribe_audio():
    print("Testing started.")
    dataset = load_dataset("common_voice", "zh-CN", split="test[:1%]")
    sample_audio_paths = [data['path'] for data in dataset]

    # Test case 1: Non-empty list of audio paths
    print("Testing case [1/3] started.")
    result = transcribe_audio(sample_audio_paths)
    assert result, f"Test case [1/3] failed: Expected transcriptions, got {result}"

    # Test case 2: Empty list of audio paths
    print("Testing case [2/3] started.")
    try:
        transcribe_audio([])
        assert False, "Test case [2/3] failed: ValueError not raised for empty audio_paths"
    except ValueError as e:
        assert str(e) == 'The list of audio paths should not be empty.', f"Test case [2/3] failed: {e}"

    # Test case 3: File not found
    print("Testing case [3/3] started.")
    try:
        transcribe_audio(['nonexistent_file.wav'])
        assert False, "Test case [3/3] failed: FileNotFoundError not raised for nonexistent file"
    except FileNotFoundError as e:
        assert 'nonexistent_file.wav' in str(e), f"Test case [3/3] failed: {e}"
    
    print("Testing finished.")

# call_test_function_line --------------------

test_transcribe_audio()