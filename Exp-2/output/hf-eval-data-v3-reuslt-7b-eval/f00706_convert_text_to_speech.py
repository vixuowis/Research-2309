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

    model_path = os.environ['MODEL_PATH']
    if model_path is None:
        raise ValueError('The environment variable MODEL_PATH must be defined and point to a valid pre-trained Chinese TTS model.')
        
    hifigan_model_file = os.path.join(model_path, 'hifigan_tacotron2.pt')
    vocoder_config_file = os.path.join(model_path, 'hifigan.yaml')
    token_list = os.path.join(model_path, 'token_list')

    text2speech = Text2Speech(
        train_config=None,
        model_file=hifigan_model_file,
        vocoder_config=vocoder_config_file,
        token_list=token_list,
        use_feats_extract=False,
        normalize_output_wav=True,
    )
    
    output = text2speech(lesson_text)['wav']
    
    with soundfile.SoundFile(output_file, 'w', samplerate=text2speech.fs, channels=1) as f:
        f.write(output.view(-1).numpy())

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