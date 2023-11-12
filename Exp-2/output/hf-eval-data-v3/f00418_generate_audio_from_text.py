# function_import --------------------

from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
import IPython.display as ipd

# function_code --------------------

def generate_audio_from_text(text):
    """
    Generate audio from Chinese text using Fairseq's pre-trained model.

    Args:
        text (str): Chinese text to be converted to audio.

    Returns:
        Audio object: An audio object that can be played in Jupyter notebooks.
    """
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        'facebook/tts_transformer-zh-cv7_css10',
        arg_overrides={'vocoder': 'hifigan', 'fp16': False}
    )
    model = models[0]
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator(model, cfg)
    sample = TTSHubInterface.get_model_input(task, text)
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
    return ipd.Audio(wav, rate=rate)

# test_function_code --------------------

def test_generate_audio_from_text():
    """
    Test the function generate_audio_from_text.
    """
    text = '你好，欢迎来到数字世界。'
    audio = generate_audio_from_text(text)
    assert isinstance(audio, ipd.Audio), 'The result should be an Audio object.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_audio_from_text()