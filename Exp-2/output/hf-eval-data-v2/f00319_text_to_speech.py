# function_import --------------------

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd

# function_code --------------------

def text_to_speech(text):
    '''
    Convert the input text to speech using the FastSpeech2 pre-trained speech synthesis model.
    
    Args:
        text (str): The input text to be converted to speech.
    
    Returns:
        An audio object that can be played in Jupyter notebooks.
    
    Raises:
        Exception: If there is an error in loading the model or generating the speech.
    '''
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub('facebook/fastspeech2-en-200_speaker-cv4', arg_overrides={'vocoder': 'hifigan', 'fp16': False})
    model = models[0]
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator(model, cfg)
    sample = TTSHubInterface.get_model_input(task, text)
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
    return ipd.Audio(wav, rate=rate)

# test_function_code --------------------

def test_text_to_speech():
    '''
    Test the text_to_speech function with a sample text.
    '''
    text = 'Hello, this is a test run.'
    audio = text_to_speech(text)
    assert isinstance(audio, ipd.Audio), 'The function should return an Audio object.'

# call_test_function_code --------------------

test_text_to_speech()