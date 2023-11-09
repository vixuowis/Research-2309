# function_import --------------------

import os
import torchaudio
from fairseq import hub_utils, checkpoint_utils
from fairseq.models.speech_to_text import S2THubInterface

# function_code --------------------

def translate_english_to_hokkien(audio_file_path):
    """
    Translates an English audio file to Hokkien using the 'facebook/xm_transformer_s2ut_en-hk' model.

    Args:
        audio_file_path (str): The path to the English audio file.

    Returns:
        hokkien_translation (torch.Tensor): The Hokkien translation of the English audio.
    """
    cache_dir = os.getenv('HUGGINGFACE_HUB_CACHE')
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task_from_hf_hub('facebook/xm_transformer_s2ut_en-hk', arg_overrides={'config_yaml': 'config.yaml', 'task': 'speech_to_text'}, cache_dir=cache_dir)
    model = models[0].cpu()
    cfg['task'].cpu = True
    generator = task.build_generator([model], cfg)

    english_audio, _ = torchaudio.load(audio_file_path)
    english_sample = S2THubInterface.get_model_input(task, english_audio)
    hokkien_translation = S2THubInterface.get_prediction(task, model, generator, english_sample)

    return hokkien_translation

# test_function_code --------------------

def test_translate_english_to_hokkien():
    """
    Tests the 'translate_english_to_hokkien' function by translating a sample English audio file and checking the type of the output.
    """
    sample_audio_file_path = 'path/to/sample/english/audio/file'
    hokkien_translation = translate_english_to_hokkien(sample_audio_file_path)

    assert isinstance(hokkien_translation, torch.Tensor), 'The output should be a torch.Tensor.'

# call_test_function_code --------------------

test_translate_english_to_hokkien()