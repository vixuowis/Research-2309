# requirements_file --------------------

import subprocess

requirements = ["fairseq"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
import IPython.display as ipd

# function_code --------------------

def synthesize_chinese_text(text):
    """
    Synthesizes Chinese text into speech using a pre-trained Fairseq TTS model.

    Args:
        text (str): The Chinese text to synthesize.

    Returns:
        An IPython.display.Audio object with the synthesized speech.

    Raises:
        RuntimeError: If the model fails to generate the speech.
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

def test_synthesize_chinese_text():
    print("Testing started.")
    test_cases = [
        ("你好，欢迎来到数字世界。", 'Output should be an IPython.display.Audio object'),
        ("今天天气不错。", 'Output should be an IPython.display.Audio object'),
        ("学习让我快乐。", 'Output should be an IPython.display.Audio object')
    ]
    for i, (text, expected) in enumerate(test_cases):
        print(f"Testing case [{i+1}/{len(test_cases)}] started.")
        output = synthesize_chinese_text(text)
        assert isinstance(output, ipd.Audio), f"Test case [{i+1}/{len(test_cases)}] failed: {expected}"
    print("Testing finished.")

# call_test_function_line --------------------

test_synthesize_chinese_text()