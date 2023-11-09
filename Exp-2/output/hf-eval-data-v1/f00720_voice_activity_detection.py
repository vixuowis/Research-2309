from pyannote.audio.core.inference import Inference


def voice_activity_detection(audio_file):
    """
    This function uses the 'julien-c/voice-activity-detection' model from Hugging Face Transformers
    to detect voice activity in an audio file and separate it from the silent parts.

    Parameters:
    audio_file (str): The path to the audio file.

    Returns:
    dict: A dictionary containing the detected voice activity segments.
    """

    # Create an Inference object by specifying the model 'julien-c/voice-activity-detection'
    model = Inference('julien-c/voice-activity-detection', device='cuda')

    # Process the audio file with the model
    result = model({
        'audio': audio_file
    })

    return result