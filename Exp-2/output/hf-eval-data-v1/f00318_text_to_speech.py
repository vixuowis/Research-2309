from speechbrain.pretrained import Tacotron2, HIFIGAN
import torchaudio

def text_to_speech(text):
    '''
    This function converts the provided text into speech using the Tacotron2 and HIFIGAN models from SpeechBrain.
    The generated speech is saved as a .wav file.
    
    Args:
    text (str): The text to be converted into speech.
    
    Returns:
    None
    '''
    # Load the pre-trained TTS model
    tacotron2 = Tacotron2.from_hparams(source='speechbrain/tts-tacotron2-ljspeech')
    # Load the pre-trained vocoder
    hifi_gan = HIFIGAN.from_hparams(source='speechbrain/tts-hifigan-ljspeech')
    # Convert the text into a spectrogram
    mel_output, mel_length, alignment = tacotron2.encode_text(text)
    # Generate the waveform from the spectrogram
    waveforms = hifi_gan.decode_batch(mel_output)
    # Save the waveform as a .wav file
    torchaudio.save('example_TTS.wav', waveforms.squeeze(1), 22050)