# requirements_file --------------------

!pip install -U speechbrain torchaudio

# function_import --------------------

import torchaudio
from speechbrain.pretrained import Tacotron2, HIFIGAN

# function_code --------------------

def text_to_speech_conversion(text):
    """
    Convert the input text to speech using SpeechBrain's pre-trained Tacotron2 and HIFIGAN models.

    Parameters:
        text (str): The text to convert to speech.

    Returns:
        str: The path to the saved WAV file containing the synthesized speech.
    """
    # Load the pre-trained Tacotron2 and HIFIGAN models
    tacotron2 = Tacotron2.from_hparams(source='speechbrain/tts-tacotron2-ljspeech')
    hifi_gan = HIFIGAN.from_hparams(source='speechbrain/tts-hifigan-ljspeech')
    
    # Convert text to spectrogram
    mel_output, mel_length, alignment = tacotron2.encode_text(text)
    
    # Convert spectrogram to audio waveform
    waveforms = hifi_gan.decode_batch(mel_output)
    
    # Save the audio waveform to a WAV file
    wav_filename = 'output_TTS.wav'
    torchaudio.save(wav_filename, waveforms.squeeze(1), 22050)
    
    return wav_filename

# test_function_code --------------------

def test_text_to_speech_conversion():
    # Define a sample text to convert
    sample_text = "The sun was shining brightly, and the birds were singing sweetly."
    
    # Perform the text to speech conversion
    wav_filename = text_to_speech_conversion(sample_text)
    
    # Check if the output WAV file exists
    assert os.path.isfile(wav_filename), f"The WAV file was not successfully created: {wav_filename}"
    
    # Check that the WAV file is not empty
    assert os.path.getsize(wav_filename) > 0, "The WAV file is empty."
    
    print("Test case passed: The text was successfully converted to speech.")