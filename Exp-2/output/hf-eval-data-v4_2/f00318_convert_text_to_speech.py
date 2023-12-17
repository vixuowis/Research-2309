# requirements_file --------------------

!pip install -U torchaudio speechbrain

# function_import --------------------

import torchaudio
from speechbrain.pretrained import Tacotron2, HIFIGAN

# function_code --------------------

def convert_text_to_speech(text: str, output_file: str) -> str:
    """
    Convert the given text into speech and save it as a .wav file.

    Args:
        text (str): The text that needs to be converted to speech.
        output_file (str): The name of the output .wav file.

    Returns:
        str: The path to the saved .wav file.

    Raises:
        RuntimeError: If any error occurs during the conversion process.
    """
    try:
        # Load the pretrained Tacotron2 and HIFIGAN models
        tacotron2 = Tacotron2.from_hparams(source='speechbrain/tts-tacotron2-ljspeech')
        hifi_gan = HIFIGAN.from_hparams(source='speechbrain/tts-hifigan-ljspeech')

        # Encode the text and generate waveforms
        mel_output, mel_length, alignment = tacotron2.encode_text(text)
        waveforms = hifi_gan.decode_batch(mel_output)

        # Save the audio to a file
        torchaudio.save(output_file, waveforms.squeeze(1), 22050)
        return output_file
    except Exception as e:
        raise RuntimeError('Error during text-to-speech conversion: ' + str(e))

# test_function_code --------------------

def test_convert_text_to_speech():
    print("Testing started.")
    # Test case 1: Convert a simple sentence
    print("Testing case [1/1] started.")
    output_file = 'example_TTS.wav'
    sample_text = 'Mary had a little lamb'
    result_path = convert_text_to_speech(sample_text, output_file)
    assert os.path.exists(result_path), f'Test case [1/1] failed: Output file {result_path} does not exist.'
    print(f'Test case [1/1] passed: Output file {result_path} exists.')
    print("Testing finished.")

# call_test_function_line --------------------

test_convert_text_to_speech()