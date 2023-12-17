# requirements_file --------------------

!pip install -U fairseq IPython

# function_import --------------------

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd

# function_code --------------------

def generate_tts_audio(message: str) -> ipd.Audio:
    """
    Generates an audio representation of the given text message using a pre-trained FastSpeech 2 model.

    Args:
        message (str): The text message to convert to speech.

    Returns:
        ipd.Audio: An IPython.display.Audio object with the generated speech audio.

    Raises:
        ValueError: If the text message is empty.
    """
    if not message:
        raise ValueError('The text message is empty.')

    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        'facebook/fastspeech2-en-ljspeech',
        arg_overrides={'vocoder': 'hifigan', 'fp16': False}
    )
    model = models[0]
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator(model, cfg)
    sample = TTSHubInterface.get_model_input(task, message)
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)

    return ipd.Audio(wav, rate=rate)

# test_function_code --------------------

def test_generate_tts_audio():
    print("Testing started.")

    # Test case 1: Non-empty message
    print("Testing case [1/1] started.")
    message = "This is a sensitive warning message. Please be aware and act accordingly."
    audio_output = generate_tts_audio(message)
    assert isinstance(audio_output, ipd.Audio), f"Test case [1/1] failed: Expected ipd.Audio object, got {type(audio_output)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_tts_audio()