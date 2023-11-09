# function_import --------------------

from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
import IPython.display as ipd

# function_code --------------------

def generate_audio_from_text(text):
    """
    This function generates an audio file from a given Chinese text using the pre-trained model 'facebook/tts_transformer-zh-cv7_css10' from Fairseq.
    
    Args:
        text (str): The Chinese text to be converted to audio.
    
    Returns:
        Audio object: An audio file generated from the input text.
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
    This function tests the 'generate_audio_from_text' function by providing a sample Chinese text and checking if the output is an instance of IPython.lib.display.Audio.
    """
    sample_text = '你好，欢迎来到数字世界。'
    output = generate_audio_from_text(sample_text)
    assert isinstance(output, ipd.lib.display.Audio), 'Output should be an instance of IPython.lib.display.Audio'

# call_test_function_code --------------------

test_generate_audio_from_text()