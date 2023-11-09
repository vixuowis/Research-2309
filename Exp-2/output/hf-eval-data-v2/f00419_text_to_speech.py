# function_import --------------------

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd

# function_code --------------------

def text_to_speech(text):
    """
    This function converts a given text into speech using FastSpeech 2 model.

    Args:
        text (str): The text to be converted into speech.

    Returns:
        Audio object: An IPython.display.Audio object that plays the converted speech.
    """
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        'facebook/fastspeech2-en-200_speaker-cv4',
        arg_overrides={'vocoder': 'hifigan', 'fp16': False}
    )
    model = models[0]
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator(model, cfg)
    sample = TTSHubInterface.get_model_input(task, text)
    wav, rate = TTSHubInterface.get_model_input(task, model, generator, sample)
    return ipd.Audio(wav, rate=rate)

# test_function_code --------------------

def test_text_to_speech():
    """
    This function tests the text_to_speech function by providing a sample text and checking the type of the output.
    """
    sample_text = 'Hello, this is a test run.'
    output = text_to_speech(sample_text)
    assert isinstance(output, ipd.Audio), 'Output should be an IPython.display.Audio object.'

# call_test_function_code --------------------

test_text_to_speech()