# requirements_file --------------------

!pip install -U pyannote.audio

# function_import --------------------

from pyannote.audio import Pipeline

# function_code --------------------

def detect_active_speech_in_audio(audio_file: str) -> List[Tuple[float, float]]:
    """
    Detects active speech intervals in an audio file.

    Args:
        audio_file (str): The path to the audio file (.wav format).

    Returns:
        List[Tuple[float, float]]: A list of tuples containing the start and end times of active speech.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the audio file is not in .wav format.
    """
    # Check if the audio file exists
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    # Check if the audio file is in .wav format
    if not audio_file.endswith('.wav'):
        raise ValueError(f"Unsupported audio format: {audio_file}")

    # Initialize the pretrained pipeline
    pipeline = Pipeline.from_pretrained('pyannote/voice-activity-detection')
    # Apply the pipeline to the audio file
    output = pipeline(audio_file)
    # Extract active speech intervals
    active_speech_intervals = [(speech.start, speech.end) for speech in output.get_timeline().support()]

    return active_speech_intervals

# test_function_code --------------------

def test_detect_active_speech_in_audio():
    print("Testing started.")
    # Assuming we have a test audio file 'test_audio.wav'
    test_audio_file = 'test_audio.wav'

    # Testing case 1: Valid .wav audio file
    print("Testing case [1/3] started.")
    try:
        result = detect_active_speech_in_audio(test_audio_file)
        assert result, f"Test case [1/3] failed: No active speech detected"
    except Exception as e:
        assert False, f"Test case [1/3] failed: {e}"

    # Testing case 2: Non-existent audio file
    print("Testing case [2/3] started.")
    try:
        result = detect_active_speech_in_audio('nonexistent_audio.wav')
        assert False, "Test case [2/3] failed: FileNotFoundError not raised"
    except FileNotFoundError:
        pass
    except Exception as e:
        assert False, f"Test case [2/3] failed: {e}"

    # Testing case 3: Unsupported audio format
    print("Testing case [3/3] started.")
    try:
        result = detect_active_speech_in_audio('unsupported_audio.mp3')
        assert False, "Test case [3/3] failed: ValueError not raised for unsupported format"
    except ValueError:
        pass
    except Exception as e:
        assert False, f"Test case [3/3] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_active_speech_in_audio()