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

def convert_text_to_speech(text):
    """Converts text to speech using FastSpeech 2 model.

    Args:
        text (str): The text to be converted into speech.

    Returns:
        An IPython.display.Audio object containing the audio data.

    Raises:
        ValueError: If the text is empty or not provided.
    """
    if not text:
        raise ValueError("Text input is empty or not provided.")

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

    # Testing case 1: Providing valid text
    print("Testing case [1/1] started.")
    text = "Hello, this is a test run."
    audio_output = convert_text_to_speech(text)
    assert isinstance(audio_output, ipd.Audio), f"Test case [1/1] failed: Expected audio output is not an instance of IPython.display.Audio"
    print("Testing finished.")

# call_test_function_line --------------------

test_convert_text_to_speech()