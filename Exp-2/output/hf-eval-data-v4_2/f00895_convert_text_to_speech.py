# requirements_file --------------------

!pip install -U speechbrain torchaudio

# function_import --------------------

import torchaudio
from speechbrain.pretrained import Tacotron2, HIFIGAN

# function_code --------------------

def convert_text_to_speech(text: str) -> str:
    """
    Convert input text to speech and save the audio file.

    Args:
        text (str): The text to be converted to speech.

    Returns:
        str: The file name of the saved audio file.

    Raises:
        ValueError: If the input text is empty.
    """
    if not text:
        raise ValueError("Input text is empty.")

    tacotron2 = Tacotron2.from_hparams(source='speechbrain/tts-tacotron2-ljspeech')
    hifi_gan = HIFIGAN.from_hparams(source='speechbrain/tts-hifigan-ljspeech')

    mel_output, mel_length, alignment = tacotron2.encode_text(text)
    waveforms = hifi_gan.decode_batch(mel_output)
    file_name = 'output_TTS.wav'
    torchaudio.save(file_name, waveforms.squeeze(1), 22050)
    return file_name

# test_function_code --------------------

def test_convert_text_to_speech():
    print("Testing started.")
    # Test case 1: Convert valid text
    print("Testing case [1/1] started.")
    result = convert_text_to_speech("The sun was shining brightly, and the birds were singing sweetly.")
    assert result == 'output_TTS.wav', f"Test case [1/1] failed: Expected 'output_TTS.wav', got {result}"
    print("Testing finished.")

# call_test_function_line --------------------

test_convert_text_to_speech()