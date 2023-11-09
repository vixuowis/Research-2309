from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd

def text_to_speech(text):
    '''
    This function converts the input text to speech using the FastSpeech2 pre-trained speech synthesis model.
    Args:
    text: str, the input text to be converted to speech.
    Returns:
    An audio file of the spoken text.
    '''
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub('facebook/fastspeech2-en-200_speaker-cv4', arg_overrides={'vocoder': 'hifigan', 'fp16': False})
    model = models[0]
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator(model, cfg)
    sample = TTSHubInterface.get_model_input(task, text)
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
    return ipd.Audio(wav, rate=rate)