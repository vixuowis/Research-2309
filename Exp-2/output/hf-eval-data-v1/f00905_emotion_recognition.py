from transformers import Wav2Vec2Model
import librosa
import numpy as np


def emotion_recognition(audio_path: str) -> np.array:
    """
    Analyze the emotion of children while they brush their teeth using a pre-trained model.

    Args:
        audio_path (str): The path to the audio file.

    Returns:
        np.array: An array of probabilities for each emotion category.
    """
    model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large-xlsr-53')
    audio_data, sample_rate = librosa.load(audio_path)
    # Process and prepare audio_data for the model
    # Use the model to analyze the emotion of the children
    return model(audio_data)