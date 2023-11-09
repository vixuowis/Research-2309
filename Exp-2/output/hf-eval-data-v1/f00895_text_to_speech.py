import torchaudio
from speechbrain.pretrained import Tacotron2, HIFIGAN

def text_to_speech(text: str, output_file: str = 'example_TTS.wav') -> None:
    """
    Converts a given text into speech and saves the output as a .wav file.

    Args:
        text (str): The text to be converted into speech.
        output_file (str, optional): The name of the output .wav file. Defaults to 'example_TTS.wav'.

    Returns:
        None
    """
    tacotron2 = Tacotron2.from_hparams(source='speechbrain/tts-tacotron2-ljspeech')
    hifi_gan = HIFIGAN.from_hparams(source='speechbrain/tts-hifigan-ljspeech')
    mel_output, mel_length, alignment = tacotron2.encode_text(text)
    waveforms = hifi_gan.decode_batch(mel_output)
    torchaudio.save(output_file, waveforms.squeeze(1), 22050)