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

    # check if all requirements for the TTS process have been installed correctly
    try:
        from espnet_model_zoo.downloader import ModelDownloader
        import torch
        import soundfile
    except Exception as e:
        raise ModuleNotFoundError("Could not find one of these modules ('espnet', 'torch', 'soundfile')! Please install them first.") from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    d = ModelDownloader()
    text2speech = Text2Speech(
        **d.download_and_unpack(
            "kan-bayashi/css10zh_nishitani_raw_phn_tacotron",
            quiet=True,
        ),
        device=device,
        # Only for Tacotron 2
        threshold=0.5,
        minlenratio=0.0,
        maxlenratio=10.0,
        use_att_constraint=False,
        backward_window=1,
        forward_window=3,
    )

    # convert the text into speech and save it as a file
    with torch.no_grad():
        wav, sr = text2speech(lesson_text)
    
    soundfile.write(output_file, wav.numpy(), samplerate=sr)

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