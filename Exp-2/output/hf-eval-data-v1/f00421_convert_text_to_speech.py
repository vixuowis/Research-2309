from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface


def convert_text_to_speech(text):
    """
    This function converts a given text to speech using the 'facebook/tts_transformer-fr-cv7_css10' model from Fairseq.
    The model is specialized in converting French text to speech.
    
    Args:
    text (str): The text to be converted to speech.
    
    Returns:
    wav (numpy array): The generated speech in the form of a wave file.
    rate (int): The sample rate of the generated speech.
    """
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