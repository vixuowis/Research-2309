from fairseq.models.wav2vec.wav2vec2_asr import Wav2Vec2Model
from huggingface_hub import cached_download


def translate_audio(input_audio):
    """
    This function translates a Czech language audio file to English language audio file using a pretrained model.
    
    Parameters:
    input_audio (str): The path to the Czech language audio file
    
    Returns:
    english_audio: The translated English language audio
    """
    # Load the pretrained model
    model = Wav2Vec2Model.from_pretrained(cached_download('https://huggingface.co/facebook/textless_sm_cs_en/resolve/main/model.pt'))
    
    # Translate the input audio to English
    english_audio = model.translate(input_audio)
    
    return english_audio