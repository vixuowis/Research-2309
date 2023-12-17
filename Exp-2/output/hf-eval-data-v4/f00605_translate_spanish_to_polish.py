# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# function_code --------------------

def translate_spanish_to_polish(spanish_text):
    """
    Translate Spanish text into Polish using pre-trained MBART model.

    Parameters:
    - spanish_text (str): A string containing Spanish text to be translated.

    Returns:
    - str: Translated text in Polish.
    """
    # Load the MBART model and tokenizer
    model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
    tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
    tokenizer.src_lang = 'es_ES'
    
    # Tokenize the Spanish text
    encoded_spanish = tokenizer(spanish_text, return_tensors='pt')
    
    # Generate the translated text in Polish
    generated_tokens = model.generate(**encoded_spanish, forced_bos_token_id=tokenizer.lang_code_to_id['pl_PL'])
    
    # Decode the generated tokens to get the Polish text
    polish_subtitles = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return polish_subtitles[0]

# test_function_code --------------------

def test_translate_spanish_to_polish():
    # Test the function translate_spanish_to_polish
    print("Testing started.")

    # Test case: translating a known Spanish sentence
    spanish_sentence = "Hola, ¿Cómo estás?"
    expected_polish_translation = "Cześć, jak się masz?"  # Expected output may vary depending on model updates

    print("Testing Spanish to Polish translation [1/1].")
    translated_polish = translate_spanish_to_polish(spanish_sentence)

    assert translated_polish == expected_polish_translation, f"Test case failed: Expected '{{expected_polish_translation}}', got '{{translated_polish}}'"

    print("Testing finished.")

# Run the test function
test_translate_spanish_to_polish()