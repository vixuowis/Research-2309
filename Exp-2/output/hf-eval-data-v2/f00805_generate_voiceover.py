# function_import --------------------

import torchaudio
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN

# function_code --------------------

def generate_voiceover(text, tmpdir_tts='./tts', tmpdir_vocoder='./vocoder'):
    """
    This function generates a voiceover from the input text using Tacotron2 and HIFIGAN models.

    Args:
        text (str): The input text to convert to voiceover.
        tmpdir_tts (str): The directory to save the Tacotron2 model. Default is './tts'.
        tmpdir_vocoder (str): The directory to save the HIFIGAN model. Default is './vocoder'.

    Returns:
        None. The function saves the generated voiceover as an audio file in the current directory.
    """
    tacotron2 = Tacotron2.from_hparams(source='padmalcom/tts-tacotron2-german', savedir=tmpdir_tts)
    hifi_gan = HIFIGAN.from_hparams(source='padmalcom/tts-hifigan-german', savedir=tmpdir_vocoder)
    mel_output, mel_length, alignment = tacotron2.encode_text(text)
    waveforms = hifi_gan.decode_batch(mel_output)
    torchaudio.save('example_TTS.wav', waveforms.squeeze(1), 22050)

# test_function_code --------------------

def test_generate_voiceover():
    """
    This function tests the generate_voiceover function by generating a voiceover from a sample text.
    """
    sample_text = 'Mary hatte ein kleines Lamm'
    generate_voiceover(sample_text)
    assert os.path.exists('example_TTS.wav'), 'Voiceover file not found.'

# call_test_function_code --------------------

test_generate_voiceover()