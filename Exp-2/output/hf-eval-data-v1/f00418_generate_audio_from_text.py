from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
import IPython.display as ipd


def generate_audio_from_text(text):
    """
    This function takes a Chinese text as input and generates an audio file in female voice.
    It uses the pre-trained model 'facebook/tts_transformer-zh-cv7_css10' from Fairseq.
    The vocoder is set to 'hifigan' and FP16 is disabled.
    
    Parameters:
    text (str): The Chinese text to be converted to audio.
    
    Returns:
    Audio: The generated audio file.
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