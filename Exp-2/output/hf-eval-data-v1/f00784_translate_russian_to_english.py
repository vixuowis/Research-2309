from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def translate_russian_to_english(russian_text):
    """
    Translates Russian text to English using the Helsinki-NLP/opus-mt-ru-en model from Hugging Face Transformers.

    Args:
        russian_text (str): The Russian text to be translated.

    Returns:
        str: The translated English text.
    """
    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ru-en')
    model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-ru-en')
    inputs = tokenizer(russian_text, return_tensors="pt")
    outputs = model.generate(**inputs)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation