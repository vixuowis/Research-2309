# requirements_file --------------------

!pip install -U huggingface_hub asteroid soundfile

# function_import --------------------

from huggingface_hub import hf_hub_download
from asteroid import ConvTasNet
from asteroid.utils.hub_utils import load_model
import soundfile as sf

# function_code --------------------

def separate_speakers(audio_path):
    """
    Separate the voices of two speakers in a single-channel audio recording.

    Parameters:
        audio_path: A string path to the single-channel audio file containing two speakers.

    Returns:
        A tuple of numpy arrays containing the separated audio signals of the two speakers.
    """
    # Download the pre-trained ConvTasNet model for voice separation
    repo_id = 'JorisCos/ConvTasNet_Libri2Mix_sepclean_8k'
    filename = hf_hub_download(repo_id, 'model.pth')
    model = load_model(filename)

    # Load the audio file
    noisy_audio, sr = sf.read(audio_path)

    # Perform separation
    separated_sources = model(noisy_audio)

    # Return the separated audio signals
    return separated_sources

# test_function_code --------------------

def test_separate_speakers():
    print("Testing started.")
    audio_file = 'path_to_audio_file.wav'  # Audio file containing voices of two speakers

    # Test: Function should return a tuple of two items for separation
    print("Testing case [1/1] started.")
    separated_sources = separate_speakers(audio_file)
    assert isinstance(separated_sources, tuple) and len(separated_sources) == 2, \
        f"Test case [1/1] failed: Expected tuple of length 2, got {len(separated_sources)}"
    print("Testing completed.")

# Run the test function
test_separate_speakers()