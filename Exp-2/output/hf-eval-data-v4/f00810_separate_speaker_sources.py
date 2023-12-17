# requirements_file --------------------

!pip install -U asteroid librosa

# function_import --------------------

import asteroid
from asteroid.models import ConvTasNet_Libri3Mix_sepclean_8k
import librosa

# function_code --------------------

def separate_speaker_sources(audio_file_path):
    '''
    Separate the speaker sources from an audio file.

    Parameters:
        audio_file_path (str): The file path to the input audio file.

    Returns:
        List: A list of numpy arrays, each representing a separated audio source.
    '''
    # Load the model
    model = ConvTasNet_Libri3Mix_sepclean_8k()
    # Load the audio
    audio, _ = librosa.load(audio_file_path, sr=None, mono=False)
    # Separate sources
    sep_sources = model.separate(audio)
    return sep_sources

# test_function_code --------------------

def test_separate_speaker_sources():
    print("Testing separate_speaker_sources function.")
    # Test with an example audio file (This file should exist for test to run)
    sep_sources = separate_speaker_sources('example_audio.wav')
    num_sources = 3  # Expected number of separated sources
    assert isinstance(sep_sources, list), "Output should be a list."
    assert len(sep_sources) == num_sources, f"Expected {num_sources} separated sources, got {len(sep_sources)}."
    for source in sep_sources:
        assert isinstance(source, np.ndarray), "Each source should be a numpy array."
    print("All tests passed.")

test_separate_speaker_sources()