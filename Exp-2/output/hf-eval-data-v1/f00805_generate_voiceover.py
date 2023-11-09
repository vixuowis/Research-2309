import torchaudio
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN

def generate_voiceover(text: str, tts_dir: str = './tts', vocoder_dir: str = './vocoder') -> None:
    """
    Generate a voiceover from the input text using Tacotron2 and HIFIGAN.

    Args:
        text (str): The input text to convert to voiceover.
        tts_dir (str, optional): The directory to save the Tacotron2 model. Defaults to './tts'.
        vocoder_dir (str, optional): The directory to save the HIFIGAN model. Defaults to './vocoder'.

    Returns:
        None. The function saves the generated voiceover as an audio file.
    """
    tacotron2 = Tacotron2.from_hparams(source='padmalcom/tts-tacotron2-german', savedir=tts_dir)
    hifi_gan = HIFIGAN.from_hparams(source='padmalcom/tts-hifigan-german', savedir=vocoder_dir)
    mel_output, _, _ = tacotron2.encode_text(text)
    waveforms = hifi_gan.decode_batch(mel_output)
    torchaudio.save('example_TTS.wav', waveforms.squeeze(1), 22050)