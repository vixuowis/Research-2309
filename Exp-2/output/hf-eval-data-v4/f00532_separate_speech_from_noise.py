# requirements_file --------------------

!pip install -U huggingface_hub torchaudio asteroid

# function_import --------------------

from huggingface_hub import hf_hub_download
import torchaudio
from asteroid.models import ConvTasNet


# function_code --------------------

def separate_speech_from_noise(audio_path):
    """
    Separates speech from background noise in an audio file.

    Parameters:
    audio_path (str): Path to the audio file to be processed.

    Returns:
    torch.Tensor: The separated speech as a PyTorch tensor.
    """
    repo_id = 'JorisCos/ConvTasNet_Libri2Mix_sepclean_8k'
    model_path = hf_hub_download(repo_id=repo_id)
    model = ConvTasNet.from_pretrained(model_path)

    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_path)

    # Perform source separation
    separated_sources = model.separate(waveform)

    # Assuming the first source is the speech
    speech = separated_sources[0]
    return speech


# test_function_code --------------------

def test_separate_speech_from_noise():
    print("Testing separate_speech_from_noise function.")
    sample_audio_path = "sample.wav"  # Replace with a path to a real audio sample

    # Test case 1: Check the type of returned result is a torch.Tensor
    print("Testing case [1/1] started.")
    separated_speech = separate_speech_from_noise(sample_audio_path)
    assert isinstance(separated_speech, torch.Tensor), f"Test case [1/1] failed: The result should be a torch.Tensor."
    print("Testing finished.")

# Run the test function
test_separate_speech_from_noise()
