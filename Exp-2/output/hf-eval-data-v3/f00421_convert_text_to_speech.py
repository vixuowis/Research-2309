# function_import --------------------

import numpy as np
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

# function_code --------------------

def convert_text_to_speech(text):
    '''
    Convert given text to speech using Fairseq's text-to-speech model.

    Args:
        text (str): The text to be converted to speech.

    Returns:
        numpy.ndarray: The generated speech in the form of a waveform.
        int: The sample rate of the generated speech.
    '''
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        'facebook/tts_transformer-fr-cv7_css10',
        arg_overrides={'vocoder': 'hifigan', 'fp16': False}
    )

    model = models[0]
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator(model, cfg)

    sample = TTSHubInterface.get_model_input(task, text)
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)

    return wav, rate

# test_function_code --------------------

def test_convert_text_to_speech():
    '''
    Test the convert_text_to_speech function.
    '''
    text = 'Bonjour, ceci est un test.'
    wav, rate = convert_text_to_speech(text)
    assert isinstance(wav, np.ndarray)
    assert isinstance(rate, int)

    text = 'Au revoir.'
    wav, rate = convert_text_to_speech(text)
    assert isinstance(wav, np.ndarray)
    assert isinstance(rate, int)

    text = 'Merci beaucoup.'
    wav, rate = convert_text_to_speech(text)
    assert isinstance(wav, np.ndarray)
    assert isinstance(rate, int)

    return 'All Tests Passed'

# call_test_function_code --------------------

test_convert_text_to_speech()