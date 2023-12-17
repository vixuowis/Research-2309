# requirements_file --------------------

!pip install -U soundfile asteroid huggingface_hub

# function_import --------------------

import soundfile as sf
from asteroid import ConvTasNet
from huggingface_hub import hf_hub_download

# function_code --------------------

def separate_speakers(audio_path):
    """
    This function uses a pre-trained ConvTasNet model to separate speakers from an audio recording.

    Args:
        audio_path (str): Path to the audio file to be processed.

    Returns:
        np.ndarray: An array of separated audio tracks for each speaker.
    """
    model_weights = hf_hub_download(repo_id='JorisCos/ConvTasNet_Libri2Mix_sepclean_8k', filename='model.pth')
    model = ConvTasNet.from_pretrained(model_weights)
    mixture_audio, sample_rate = sf.read(audio_path)
    est_sources = model.separate(mixture_audio)
    return est_sources

# test_function_code --------------------

def test_separate_speakers():
    print("Testing separate_speakers function.")
    mixture_audio, _ = sf.read("test_audio.wav")
    est_sources = separate_speakers("test_audio.wav")
    assert est_sources.shape[0] > 1, "The function did not separate the speakers correctly."
    print("Testing completed successfully.")

test_separate_speakers()