# function_import --------------------

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# function_code --------------------

def translate_spanish_to_polish(spanish_text):
    """
    Translate Spanish text to Polish using Hugging Face's MBartForConditionalGeneration model.

    Args:
        spanish_text (str): The Spanish text to be translated.

    Returns:
        str: The translated Polish text.
    """

    # Load the model and tokenizer
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50", force_download=True)
    tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50", src_lang="es_XX", tgt_lang="pl_PL", add_prefix_space=True,
    )

    # Tokenize the input text
    tokenized_text = tokenizer.prepare_seq2seq_batch([spanish_text])

    # Get the generated translation
    translated = model.generate(**tokenized_text)

    return tokenizer.postprocess_translation(translated, skip_special_tokens=True)[0]


# test_function_code --------------------

def test_translate_spanish_to_polish():
    """
    Test the function translate_spanish_to_polish.
    """
    spanish_text = 'Hola, ¿cómo estás?'
    polish_text = translate_spanish_to_polish(spanish_text)
    assert isinstance(polish_text, str), 'The result should be a string.'
    assert polish_text != '', 'The result should not be an empty string.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_translate_spanish_to_polish()