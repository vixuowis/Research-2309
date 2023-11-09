from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM

def translate_english_to_french(english_text):
    """
    This function translates English text to French using the 'optimum/t5-small' model from Transformers.

    Args:
        english_text (str): The English text to be translated.

    Returns:
        str: The translated French text.
    """
    tokenizer = AutoTokenizer.from_pretrained('optimum/t5-small')
    model = ORTModelForSeq2SeqLM.from_pretrained('optimum/t5-small')
    translator = pipeline('translation_en_to_fr', model=model, tokenizer=tokenizer)
    french_translation = translator(english_text)
    return french_translation[0]['translation_text']