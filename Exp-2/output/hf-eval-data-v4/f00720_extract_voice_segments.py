# requirements_file --------------------

!pip install -U pyannote.audio

# function_import --------------------

from pyannote.audio.core.inference import Inference

# function_code --------------------

def extract_voice_segments(audio_file_path, model_name='julien-c/voice-activity-detection', device='cuda'):
    """
    Extract segments with voice activity from an audio file using a pre-trained model.

    Parameters:
    audio_file_path (str): The path to the audio file.
    model_name (str): Name of the pre-trained model from Hugging Face Transformers.
    device (str): Device to use for inference; 'cuda' or 'cpu'.

    Returns:
    list: A list of tuples representing the segments with voice activity.
    """
    # Load the pre-trained model
    model = Inference(model_name, device=device)

    # Perform inference to find voice activity segments
    results = model({'audio': audio_file_path})

    # Extract voice activity segments
    segments = [(segment['start'], segment['end']) for segment in results]
    return segments

# test_function_code --------------------

def test_extract_voice_segments():
    print('Testing extract_voice_segments function...')

    # Example audio file path
    sample_audio_path = 'test_audio.wav'

    # Test case with the default model and device
    print('Test case with default parameters started.')
    segments = extract_voice_segments(sample_audio_path)
    assert segments, f"Test case failed: no segments detected"

    print('Testing completed successfully.')

# Run the test function
if __name__ == '__main__':
    test_extract_voice_segments()