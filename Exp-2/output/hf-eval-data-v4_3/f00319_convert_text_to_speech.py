# requirements_file --------------------

import subprocess

requirements = ["fairseq", "IPython"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd

# function_code --------------------

def convert_text_to_speech(text: str) -> IPython.display.Audio:
    """Convert given text to speech using pre-trained FastSpeech2 model.

    Args:
        text (str): The text to be converted to speech.

    Returns:
        IPython.display.Audio: An Audio object that can play the synthesized speech.

    Raises:
        ValueError: If the text is empty.
    """
    if not text:
        raise ValueError('Text for speech synthesis cannot be empty.')
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        'facebook/fastspeech2-en-200_speaker-cv4', 
        arg_overrides={'vocoder': 'hifigan', 'fp16': False}
    )
    model = models[0]
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator(model, cfg)
    sample = TTSHubInterface.get_model_input(task, text)
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
    return ipd.Audio(wav, rate=rate)

# test_function_code --------------------

def test_convert_text_to_speech():
    print("Testing started.")

    # Test case 1: Non-empty text
    print("Testing case [1/1] started.")
    result_audio = convert_text_to_speech('Hello, World!')
    assert isinstance(result_audio, IPython.display.Audio), 'Test case [1/1] failed: The result should be an Audio object.'
    print('Testing finished.')

# call_test_function_line --------------------

test_convert_text_to_speech()