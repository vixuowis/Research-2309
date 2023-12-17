# requirements_file --------------------

!pip install -U huggingsound

# function_import --------------------

from huggingsound import SpeechRecognitionModel

# function_code --------------------

def transcribe_chinese_podcasts(audio_paths):
    """
    Transcribes an array of Chinese audio files using a pre-trained Wav2Vec2 model.

    Args:
        audio_paths (list of str): The paths to the audio files.

    Returns:
        list of str: The list of transcriptions for the provided audio files.

    Raises:
        ValueError: If any provided audio path is invalid.
    """
    model = SpeechRecognitionModel('jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn')
    try:
        transcriptions = model.transcribe(audio_paths)
    except Exception as e:
        raise ValueError(f"Error processing audio paths: {str(e)}")
    return transcriptions


# test_function_code --------------------

def test_transcribe_chinese_podcasts():
    print("Testing started.")
    # Assuming we have a function `load_dataset` that gives us test audio paths
    dataset = load_dataset("chinese_podcasts")
    audio_paths = [item['audio_path'] for item in dataset]

    # Test case 1: Valid audio files
    print("Testing case [1/3] started.")
    try:
        transcriptions = transcribe_chinese_podcasts(audio_paths)
        assert len(transcriptions) == len(audio_paths), "Length of transcriptions does not match number of audio files."
        print("Testing case [1/3] finished.")
    except ValueError as ve:
        print(f"Test case [1/3] failed: {ve}")

    # Test case 2: Empty audio_paths list
    print("Testing case [2/3] started.")
    try:
        transcribe_chinese_podcasts([])
        print("Testing case [2/3] finished.")
    except ValueError:
        assert True, "Test case [2/3] failed: No exception raised for empty audio_paths list."

    # Test case 3: Invalid audio path
    print("Testing case [3/3] started.")
    invalid_audio_path = ["/invalid/path.mp3"]
    try:
        transcribe_chinese_podcasts(invalid_audio_path)
        print("Testing case [3/3] failed: Invalid audio path did not raise an exception.")
    except ValueError:
        assert True, "Should raise ValueError for invalid audio path"

    print("Testing finished.")


# call_test_function_line --------------------

test_transcribe_chinese_podcasts()