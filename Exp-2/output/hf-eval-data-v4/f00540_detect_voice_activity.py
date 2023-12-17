# requirements_file --------------------

!pip install -U pyannote.audio==2.1.1 librosa

# function_import --------------------

from pyannote.audio import Model
import librosa

# function_code --------------------

def detect_voice_activity(audio_file):
    """
    Detects voice activity in an audio file.

    Parameters:
    audio_file (str): The path to the audio file to be processed.

    Returns:
    Tuple[np.ndarray, np.ndarray]: The timestamps of the audio where voice activity was detected.
    """
    # Load the pre-trained model
    model = Model.from_pretrained('popcornell/pyannote-segmentation-chime6-mixer6')

    # Load the audio file
    signal, sample_rate = librosa.load(audio_file, sr=None)

    # Perform voice activity detection
    voice_activity = model({'audio': signal, 'sample_rate': sample_rate})

    # Extract timestamps where voice is detected
    voice_segments = [(segment.start, segment.end) for segment in voice_activity.iter_segments()]

    return voice_segments

# test_function_code --------------------

def test_detect_voice_activity():
    # Load a sample audio file (replace 'sample.wav' with an actual file path)
    audio_file = 'sample.wav'

    # Detect voice activity
    voice_segments = detect_voice_activity(audio_file)

    # Check that voice_segments is not empty
    assert voice_segments, 'No voice activity was detected.'

    # For a proper test, one should use a ground truth annotation
    # to compare against but this is out of scope for this example.
    print('Test function passed.')

# Run the test function
test_detect_voice_activity()