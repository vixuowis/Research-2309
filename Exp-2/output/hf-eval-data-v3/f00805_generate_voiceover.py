# function_import --------------------

import torchaudio
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN

# function_code --------------------

def generate_voiceover(text, tmpdir_tts='./tts', tmpdir_vocoder='./vocoder'):
    """
    Generate voiceover from text using Tacotron2 and HIFIGAN from speechbrain.

    Args:
        text (str): The text to convert to voiceover.
        tmpdir_tts (str, optional): The directory to save Tacotron2 model. Defaults to './tts'.
        tmpdir_vocoder (str, optional): The directory to save HIFIGAN model. Defaults to './vocoder'.

    Returns:
        str: The path of the generated voiceover audio file.
    """
    tacotron2 = Tacotron2.from_hparams(source='padmalcom/tts-tacotron2-german', savedir=tmpdir_tts)
    hifi_gan = HIFIGAN.from_hparams(source='padmalcom/tts-hifigan-german', savedir=tmpdir_vocoder)
    mel_output, mel_length, alignment = tacotron2.encode_text(text)
    waveforms = hifi_gan.decode_batch(mel_output)
    audio_file_path = 'example_TTS.wav'
    torchaudio.save(audio_file_path, waveforms.squeeze(1), 22050)
    return audio_file_path

# test_function_code --------------------

def test_generate_voiceover():
    """
    Test the function generate_voiceover.
    """
    # Test case 1: Normal case
    text1 = 'Mary hatte ein kleines Lamm'
    assert generate_voiceover(text1) == 'example_TTS.wav'
    # Test case 2: Empty string
    text2 = ''
    assert generate_voiceover(text2) == 'example_TTS.wav'
    # Test case 3: Long string
    text3 = 'Mary hatte ein kleines Lamm' * 100
    assert generate_voiceover(text3) == 'example_TTS.wav'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_voiceover()