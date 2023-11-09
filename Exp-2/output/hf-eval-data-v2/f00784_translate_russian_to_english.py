# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def translate_russian_to_english(text):
    """
    Translates Russian text to English using the Helsinki-NLP/opus-mt-ru-en model from Hugging Face Transformers.

    Args:
        text (str): The Russian text to be translated.

    Returns:
        str: The translated English text.
    """
    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ru-en')
    model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-ru-en')
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model.generate(**inputs)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# test_function_code --------------------

def test_translate_russian_to_english():
    """
    Tests the translate_russian_to_english function by translating a sample Russian text and checking if the output is a non-empty string.
    """
    sample_text = 'русский текст'
    translation = translate_russian_to_english(sample_text)
    assert isinstance(translation, str)
    assert len(translation) > 0

# call_test_function_code --------------------

test_translate_russian_to_english()