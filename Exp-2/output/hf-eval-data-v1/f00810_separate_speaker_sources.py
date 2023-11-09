import asteroid
from asteroid.models import ConvTasNet_Libri3Mix_sepclean_8k
import librosa


def separate_speaker_sources(audio_file_path):
    """
    This function separates the speaker sources from a given audio file using the ConvTasNet_Libri3Mix_sepclean_8k model from the Asteroid package.

    Args:
        audio_file_path (str): The path to the audio file to be processed.

    Returns:
        sep_sources (numpy.ndarray): The separated speaker sources as a numpy array.
    """
    model = ConvTasNet_Libri3Mix_sepclean_8k()
    audio, _ = librosa.load(audio_file_path, sr=None, mono=False)
    sep_sources = model.separate(audio)
    return sep_sources