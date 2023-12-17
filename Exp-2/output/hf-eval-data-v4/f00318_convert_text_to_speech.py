# requirements_file --------------------

!pip install -U speechbrain torchaudio

# function_import --------------------

from speechbrain.pretrained import Tacotron2, HIFIGAN
import torchaudio

# function_code --------------------

def convert_text_to_speech(text, output_path):
    """
    Convert the given text to speech audio file using SpeechBrain's TTS models.

    Args:
        text (str): The text to be converted to speech.
        output_path (str): Path to save the generated audio file.

    Returns:
        str: Path to the generated audio file.
    """
    # Load pre-trained TTS and vocoder models from SpeechBrain
    tacotron2 = Tacotron2.from_hparams(source='speechbrain/tts-tacotron2-ljspeech')
    hifi_gan = HIFIGAN.from_hparams(source='speechbrain/tts-hifigan-ljspeech')

    # Convert text to waveform
    mel_output, mel_length, alignment = tacotron2.encode_text(text)
    waveforms = hifi_gan.decode_batch(mel_output)

    # Save the waveform as an audio file
    torchaudio.save(output_path, waveforms.squeeze(1), 22050)

    return output_path

# test_function_code --------------------

def test_convert_text_to_speech():
    print("Testing convert_text_to_speech function.")

    # Define a sample text
    sample_text = "Mary had a little lamb."
    output_path = 'test_TTS.wav'

    # Expected output is not None and a file path
    generated_path = convert_text_to_speech(sample_text, output_path)
    assert generated_path is not None, "The function did not return a path."
    assert os.path.exists(generated_path), f"Audio file was not created at {generated_path}."
    print("Test successful.")

# Run the test function
test_convert_text_to_speech()