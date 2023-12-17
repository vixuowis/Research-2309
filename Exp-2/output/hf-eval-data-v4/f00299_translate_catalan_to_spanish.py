# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import MarianMTModel, MarianTokenizer

# function_code --------------------

def translate_catalan_to_spanish(catalan_text):
    """
    Translate Catalan text to Spanish using the pre-trained MarianMT model.

    Parameters:
        catalan_text (str): The Catalan text to be translated.

    Returns:
        str: The translated Spanish text.
    """
    model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ca-es')
    tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ca-es')
    tokenized_text = tokenizer.encode(catalan_text, return_tensors="pt")
    translated_tokens = model.generate(tokenized_text)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

# test_function_code --------------------

def test_translate_catalan_to_spanish():
    print("Testing started.")
    # Test case 1: A simple sentence
    catalan_sentence = "Hola, com estàs?"
    expected_spanish = "Hola, ¿cómo estás?"
    translated_spanish = translate_catalan_to_spanish(catalan_sentence)
    assert translated_spanish == expected_spanish, f"Test case failed: translated '{catalan_sentence}' to '{translated_spanish}' instead of '{expected_spanish}'"

    # Test case 2: A complex sentence
    catalan_sentence = "La tecnologia avança a un ritme vertiginós."
    expected_spanish = "La tecnología avanza a un ritmo vertiginoso."
    translated_spanish = translate_catalan_to_spanish(catalan_sentence)
    assert translated_spanish == expected_spanish, "Test case failed: translated '{catalan_sentence}' to '{translated_spanish}' instead of '{expected_spanish}'"

    print("Testing finished.")

# Run test function
test_translate_catalan_to_spanish()