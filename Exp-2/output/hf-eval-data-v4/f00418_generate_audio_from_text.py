# requirements_file --------------------

!pip install -U fairseq ipython 

# function_import --------------------

from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
import IPython.display as ipd

# function_code --------------------

def generate_audio_from_text(text):
    """
    Generate an audio file from Chinese text using a pre-trained model.

    :param text: Chinese text to be converted to speech.
    :return: Audio object that can be played in an IPython environment.
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
    print("Testing started.")
    test_text = '你好，欢迎来到数字世界。'  # '你好，欢迎来到数字世界。'

    # Test case 1
    print("Testing case [1/1] started.")
    audio = generate_audio_from_text(test_text)
    assert isinstance(audio, ipd.Audio), f"Test case [1/1] failed: Expected ipd.Audio instance, got {type(audio)}"
    print("Testing finished.")

test_generate_audio_from_text()