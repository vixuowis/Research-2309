from pyannote.audio.core.inference import Inference


def detect_voice_activity(audio_file):
    """
    This function uses the 'julien-c/voice-activity-detection' model from Hugging Face Transformers
    to detect voice activity in an audio file.

    Parameters:
    audio_file (str): Path to the audio file.

    Returns:
    dict: A dictionary containing the detected voice activity regions in the audio data.
    """
    # Create an instance of the Inference class with the 'julien-c/voice-activity-detection' model
    model = Inference('julien-c/voice-activity-detection', device='cuda')

    # Use the model to detect voice activity in the audio file
    voice_activity_detection_result = model({'audio': audio_file})

    return voice_activity_detection_result