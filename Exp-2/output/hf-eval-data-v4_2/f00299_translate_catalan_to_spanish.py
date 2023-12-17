# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import MarianMTModel, MarianTokenizer

# function_code --------------------

def translate_catalan_to_spanish(catalan_text: str) -> str:
    """
    Translates a given text from Catalan to Spanish using a pre-trained model.

    Args:
        catalan_text (str): The text in Catalan to be translated.

    Returns:
        str: The translated text in Spanish.

    Raises:
        ValueError: If catalan_text is not a str.
    """
    if not isinstance(catalan_text, str):
        raise ValueError('The input text must be a string.')

    # Load the pre-trained model and corresponding tokenizer
    model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ca-es')
    tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ca-es')

    # Tokenize the input text and translate it
    tokenized_text = tokenizer.encode(catalan_text, return_tensors="pt")
    translated_tokens = model.generate(tokenized_text)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    return translated_text

# test_function_code --------------------

def test_translate_catalan_to_spanish():
    print("Testing started.")
    # We will just test the function using manual strings since we are not using a dataset

    # Test case 1: Standard Catalan text
    print("Testing case [1/3] started.")
    catalan_text = "Aquesta es una prova de traduccio."
    translated_text = translate_catalan_to_spanish(catalan_text)
    assert translated_text == "Esta es una prueba de traducción.", f"Test case [1/3] failed: {translated_text}"

    # Test case 2: Empty text
    print("Testing case [2/3] started.")
    catalan_text = ""
    translated_text = translate_catalan_to_spanish(catalan_text)
    assert translated_text == "", f"Test case [2/3] failed: {translated_text}"

    # Test case 3: Non-string input
    print("Testing case [3/3] started.")
    catalan_text = 123
    try:
        _ = translate_catalan_to_spanish(catalan_text)
        assert False, "Test case [3/3] failed: No ValueError raised for non-string input."
    except ValueError as e:
        assert str(e) == "The input text must be a string.", f"Test case [3/3] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_catalan_to_spanish()