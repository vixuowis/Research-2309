# function_import --------------------

import numpy as np
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

# function_code --------------------

def text_to_speech(text):
    '''
    Translates the given text to speech for a French audiobook assistant.

    Args:
        text (str): The text to be converted to speech.

    Returns:
        numpy.ndarray: The audio output that translates the given text to speech.
        int: The sample rate of the audio output.
    '''
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub('facebook/tts_transformer-fr-cv7_css10')
    model = models[0]
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator(model, cfg)
    sample = TTSHubInterface.get_model_input(task, text)
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
    return wav, rate

# test_function_code --------------------

def test_text_to_speech():
    '''
    Tests the text_to_speech function.
    '''
    text = 'Bonjour, ceci est un test.'
    wav, rate = text_to_speech(text)
    assert isinstance(wav, np.ndarray)
    assert isinstance(rate, int)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_text_to_speech()