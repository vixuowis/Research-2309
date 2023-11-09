# function_import --------------------

from speechbrain.pretrained import Tacotron2, HIFIGAN
import torchaudio

# function_code --------------------

def text_to_speech(text: str, output_file: str = 'example_TTS.wav'):
    """
    Converts the provided text into speech and saves the generated audio into a file.

    Args:
        text (str): The text to be converted into speech.
        output_file (str): The name of the file where the generated audio will be saved. Defaults to 'example_TTS.wav'.

    Returns:
        None
    """
    tacotron2 = Tacotron2.from_hparams(source='speechbrain/tts-tacotron2-ljspeech')
    hifi_gan = HIFIGAN.from_hparams(source='speechbrain/tts-hifigan-ljspeech')
    mel_output, mel_length, alignment = tacotron2.encode_text(text)
    waveforms = hifi_gan.decode_batch(mel_output)
    torchaudio.save(output_file, waveforms.squeeze(1), 22050)

# test_function_code --------------------

def test_text_to_speech():
    """
    Tests the text_to_speech function by converting a sample text into speech and saving the generated audio into a file.

    Returns:
        None
    """
    sample_text = 'Mary had a little lamb'
    text_to_speech(sample_text, 'test_TTS.wav')
    assert os.path.exists('test_TTS.wav')

# call_test_function_code --------------------

test_text_to_speech()