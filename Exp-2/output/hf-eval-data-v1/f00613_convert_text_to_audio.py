from huggingface_hub import unit
from fairseq import TTS


def convert_text_to_audio(book_text: str, model_name: str = 'facebook/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10') -> None:
    """
    This function converts the given text into an audio file using a pre-trained Text-to-Speech model.
    
    Args:
    book_text (str): The text content of the book to be converted into an audio file.
    model_name (str, optional): The name of the pre-trained TTS model. Defaults to 'facebook/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10'.
    
    Returns:
    None
    """
    # Load the pre-trained TTS model
    model = unit.TTS.from_pretrained(model_name)
    
    # Generate the audio waveform from the book text
    waveform = model.generate_audio(book_text)
    
    # Save the generated waveform as an audio file
    waveform.save('audiobook_output.wav')