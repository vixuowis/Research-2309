from fairseq import hub_utils, checkpoint_utils
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
import torchaudio

def translate_hokkien_to_english(audio_file_path):
    """
    This function translates Hokkien speech to English text using the Fairseq's xm_transformer_s2ut_hk-en model.
    
    Parameters:
    audio_file_path (str): The path to the audio file to be translated.
    
    Returns:
    str: The translated English text.
    """
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task_from_hf_hub('facebook/xm_transformer_s2ut_hk-en', task='speech_to_text', cache_dir='./models')
    model = models[0].cpu()
    audio, _ = torchaudio.load(audio_file_path)
    generator = task.build_generator([model], cfg)
    sample = S2THubInterface.get_model_input(task, audio)
    translation = S2THubInterface.get_prediction(task, model, generator, sample)
    return translation