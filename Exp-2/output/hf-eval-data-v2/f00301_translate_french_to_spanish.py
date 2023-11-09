# function_import --------------------

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# function_code --------------------

def translate_french_to_spanish(input_text):
    """
    Translates French text to Spanish using the Helsinki-NLP/opus-mt-fr-es model from Hugging Face Transformers.

    Args:
        input_text (str): The French text to be translated.

    Returns:
        str: The translated Spanish text.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-fr-es')
    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-es')
    tokenized_input = tokenizer(input_text, return_tensors='pt')
    translated = model.generate(**tokenized_input)
    output_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return output_text

# test_function_code --------------------

def test_translate_french_to_spanish():
    """
    Tests the translate_french_to_spanish function by translating a French sentence and checking if the output is a string.
    """
    input_text = 'Bonjour, comment ça va?'
    output_text = translate_french_to_spanish(input_text)
    assert isinstance(output_text, str), 'The output should be a string.'

# call_test_function_code --------------------

test_translate_french_to_spanish()