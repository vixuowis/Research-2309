# function_import --------------------

from huggingface_hub import hf_hub_download
import soundfile as sf
from asteroid import ConvTasNet
from asteroid.models import BaseModel

# function_code --------------------

def separate_speakers(audio_file: str, model_repo_id: str = 'JorisCos/ConvTasNet_Libri2Mix_sepclean_8k') -> list:
    '''
    Separate the speakers from an audio file using the pre-trained ConvTasNet_Libri2Mix_sepclean_8k model from Hugging Face.

    Args:
        audio_file (str): The path to the audio file to be processed.
        model_repo_id (str): The repository ID of the pre-trained model on Hugging Face. Default is 'JorisCos/ConvTasNet_Libri2Mix_sepclean_8k'.

    Returns:
        list: A list of numpy arrays, each representing the audio data of a separated speaker.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the audio file is not a valid audio file.
    '''
    # Download the pre-trained model
    model_path = hf_hub_download(repo_id=model_repo_id)

    # Load the model
    model = BaseModel.from_pretrained(model_path)

    # Load the audio file
    mix, _ = sf.read(audio_file)

    # Perform source separation
    est_sources = model.separate(mix)

    return est_sources

# test_function_code --------------------

def test_separate_speakers():
    '''
    Test the separate_speakers function.
    '''
    # Test with a mono audio file
    mono_audio_file = 'mono_audio.wav'
    mono_est_sources = separate_speakers(mono_audio_file)
    assert len(mono_est_sources) == 1, 'The number of separated sources should be 1 for a mono audio file.'

    # Test with a stereo audio file
    stereo_audio_file = 'stereo_audio.wav'
    stereo_est_sources = separate_speakers(stereo_audio_file)
    assert len(stereo_est_sources) == 2, 'The number of separated sources should be 2 for a stereo audio file.'

    # Test with a non-existent audio file
    try:
        separate_speakers('non_existent.wav')
    except FileNotFoundError:
        pass
    else:
        assert False, 'A FileNotFoundError should be raised if the audio file does not exist.'

    # Test with an invalid audio file
    try:
        separate_speakers('invalid.wav')
    except ValueError:
        pass
    else:
        assert False, 'A ValueError should be raised if the audio file is not a valid audio file.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_separate_speakers()