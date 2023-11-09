# function_import --------------------

import asteroid
from asteroid.models import ConvTasNet_Libri3Mix_sepclean_8k
import librosa
import soundfile as sf

# function_code --------------------

def separate_speaker_sources(audio_file_path):
    """
    This function separates the speaker sources from the original audio file using the ConvTasNet_Libri3Mix_sepclean_8k model.

    Args:
        audio_file_path (str): The path to the audio file to be processed.

    Returns:
        sep_sources (np.ndarray): The separated speaker sources.
    """
    model = ConvTasNet_Libri3Mix_sepclean_8k()
    audio, _ = librosa.load(audio_file_path, sr=None, mono=False)
    sep_sources = model.separate(audio)
    return sep_sources

# test_function_code --------------------

def test_separate_speaker_sources():
    """
    This function tests the separate_speaker_sources function by loading a sample audio file and checking the output.
    """
    sep_sources = separate_speaker_sources('sample_audio.wav')
    assert sep_sources is not None, 'The function did not return any output.'
    assert isinstance(sep_sources, np.ndarray), 'The output is not an np.ndarray.'
    assert sep_sources.shape[0] > 0, 'The output does not contain any sources.'

# call_test_function_code --------------------

test_separate_speaker_sources()