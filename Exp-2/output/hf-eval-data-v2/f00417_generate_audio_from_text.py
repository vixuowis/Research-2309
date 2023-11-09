# function_import --------------------

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd

# function_code --------------------

def generate_audio_from_text(warning_message):
    """
    This function generates an audio waveform from a given text using the FastSpeech 2 model.

    Args:
        warning_message (str): The text to be converted to speech.

    Returns:
        audio_output: An audio object that can be played back.
    """
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub('facebook/fastspeech2-en-ljspeech', arg_overrides={'vocoder': 'hifigan', 'fp16': False})
    model = models[0]
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator(model, cfg)
    sample = TTSHubInterface.get_model_input(task, warning_message)
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
    audio_output = ipd.Audio(wav, rate=rate)
    return audio_output

# test_function_code --------------------

def test_generate_audio_from_text():
    """
    This function tests the generate_audio_from_text function by passing a sample text and checking the type of the output.
    """
    sample_text = 'This is a test message.'
    audio_output = generate_audio_from_text(sample_text)
    assert isinstance(audio_output, ipd.Audio), 'The output should be an audio object.'

# call_test_function_code --------------------

test_generate_audio_from_text()