# requirements_file --------------------

!pip install -U fairseq

# function_import --------------------

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

# function_code --------------------

def convert_text_to_speech_fr(text):
    """
    Convert French text to speech using a pre-trained Fairseq TTS model.

    Args:
        text (str): The French text to be converted to speech.

    Returns:
        Tuple[bytes, int]: A tuple containing the WAV audio bytes and the sample rate.

    Raises:
        ValueError: If the text is empty or not provided.
    """
    if not text:
        raise ValueError('Text is empty or not provided')

    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        'facebook/tts_transformer-fr-cv7_css10',
        arg_overrides={'vocoder': 'hifigan', 'fp16': False}
    )
    
    model = models[0]
    
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator(model, cfg)
    
    sample = TTSHubInterface.get_model_input(task, text)
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)

    return wav, rate

# test_function_code --------------------

def test_convert_text_to_speech_fr():
    print("Testing started.")

    # Testing case 1: Non-empty text
    print("Testing case [1/2] started.")
    text = "Bonjour, ceci est un test."
    wav, rate = convert_text_to_speech_fr(text)
    assert wav is not None and rate == 22050, f"Test case [1/2] failed: Expected non-empty WAV and rate 22050, got {rate}"

    # Testing case 2: Empty text
    print("Testing case [2/2] started.")
    try:
        _ = convert_text_to_speech_fr("")
        assert False, "Test case [2/2] failed: ValueError was not raised for empty text"
    except ValueError:
        pass
    
    print("Testing finished.")

# call_test_function_line --------------------

test_convert_text_to_speech_fr()