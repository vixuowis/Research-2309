# function_import --------------------

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

# function_code --------------------

def text_to_speech(text):
    """
    This function translates the given text to speech for a French audiobook assistant.
    
    Args:
        text (str): The text to be converted to speech.
    
    Returns:
        wav (numpy array): The audio output in the form of a numpy array.
        rate (int): The sample rate of the audio output.
    
    Raises:
        Exception: If the text is not a string.
    """
    if not isinstance(text, str):
        raise Exception('The text should be a string.')
    
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub('facebook/tts_transformer-fr-cv7_css10')
    model = models[0]
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator(model, cfg)
    sample = TTSHubInterface.get_model_input(task, text)
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
    
    return wav, rate

# test_function_code --------------------

def test_text_to_speech():
    """
    This function tests the text_to_speech function.
    
    Raises:
        Exception: If the test fails.
    """
    text = 'Bonjour, ceci est un test.'
    wav, rate = text_to_speech(text)
    
    if not isinstance(wav, np.ndarray) or not isinstance(rate, int):
        raise Exception('The test failed.')

# call_test_function_code --------------------

test_text_to_speech()