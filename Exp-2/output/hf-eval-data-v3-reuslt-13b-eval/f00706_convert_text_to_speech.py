# function_import --------------------

import os
import soundfile
from espnet2.bin.tts_inference import Text2Speech

# function_code --------------------

def convert_text_to_speech(lesson_text: str, output_file: str) -> None:
    '''
    Convert the given text into speech using a pre-trained Chinese Text-to-Speech model.

    Args:
        lesson_text (str): The text content of the lesson.
        output_file (str): The path to the output audio file.

    Returns:
        None

    Raises:
        ModuleNotFoundError: If the required modules are not installed.
    '''
    
    if 'TTS_MODEL' in os.environ == False:
        raise ModuleNotFoundError('Please set environment variable for the TTS model.')

    tts = Text2Speech.from_pretrained(os.environ['TTS_MODEL'])
    speech, sampling_rate = tts(lesson_text)
    
    soundfile.write(output_file, speech[0].numpy().astype('int16'), sampling_rate)

# test_function_code --------------------

def test_convert_text_to_speech():
    '''
    Test the convert_text_to_speech function.
    '''
    convert_text_to_speech('汉语很有趣', 'lesson_audio_example.wav')
    assert os.path.exists('lesson_audio_example.wav'), 'The audio file does not exist.'
    os.remove('lesson_audio_example.wav')
    assert not os.path.exists('lesson_audio_example.wav'), 'The audio file was not deleted.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_convert_text_to_speech()