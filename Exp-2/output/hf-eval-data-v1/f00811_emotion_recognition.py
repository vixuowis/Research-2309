from transformers import pipeline
import librosa


def emotion_recognition(audio_file_path: str):
    """
    Analyze the emotions expressed in a user's recorded message using Hugging Face Transformers.

    Args:
        audio_file_path (str): The path to the audio file.

    Returns:
        dict: The top predicted emotions.
    """
    classifier = pipeline('audio-classification', model='superb/hubert-large-superb-er')
    predicted_emotions = classifier(audio_file_path, top_k=5)
    return predicted_emotions