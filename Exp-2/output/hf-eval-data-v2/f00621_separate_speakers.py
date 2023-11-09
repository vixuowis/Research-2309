# function_import --------------------

import soundfile as sf
from asteroid import ConvTasNet
from huggingface_hub import hf_hub_download

# function_code --------------------

def separate_speakers(audio_file):
    """
    This function separates speakers from a recorded audio using the ConvTasNet_Libri2Mix_sepclean_8k model from Hugging Face Transformers.

    Args:
        audio_file (str): The path to the audio file to be processed.

    Returns:
        numpy.ndarray: A 2D array where each row corresponds to a separated speaker.
    """
    model_weights = hf_hub_download(repo_id='JorisCos/ConvTasNet_Libri2Mix_sepclean_8k', filename='model.pth')
    model = ConvTasNet.from_pretrained(model_weights)
    mixture_audio, _ = sf.read(audio_file)
    return model.separate(mixture_audio)

# test_function_code --------------------

def test_separate_speakers():
    """
    This function tests the separate_speakers function by separating speakers from a sample audio file.
    """
    separated_speakers = separate_speakers('sample_audio.wav')
    assert separated_speakers.shape[0] > 1, 'The function should separate at least two speakers.'

# call_test_function_code --------------------

test_separate_speakers()