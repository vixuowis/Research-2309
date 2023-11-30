# function_import --------------------

import os
import torchaudio
from speechbrain.pretrained import Tacotron2, HIFIGAN

# function_code --------------------

def text_to_speech(text: str, output_file: str):
    '''
    Convert the input text into speech and save the audio to a .wav file.

    Args:
        text (str): The input text to be converted into speech.
        output_file (str): The path of the output .wav file.

    Returns:
        None
    '''
    
    # TTS model
    tts_model = HIFIGAN.from_hparams(source="speechbrain/hi-fi-gan")

    # Generate speech and save to a .wav file
    result = tts_model.generate_string(text, output_file)

# __name__ --------------------

if __name__ == "__main__":

    text = "Hello world!"
    output_file = os.path.join("assets", "output.wav")

    # Convert the input text into speech and save the audio to a .wav file
    text_to_speech(text, output_file)


# test_function_code --------------------

def test_text_to_speech():
    '''
    Test the text_to_speech function.
    '''
    text_to_speech('The sun was shining brightly, and the birds were singing sweetly.', 'test_TTS.wav')
    assert os.path.exists('test_TTS.wav'), 'The output file does not exist.'
    os.remove('test_TTS.wav')
    return 'All Tests Passed'


# call_test_function_code --------------------

test_text_to_speech()