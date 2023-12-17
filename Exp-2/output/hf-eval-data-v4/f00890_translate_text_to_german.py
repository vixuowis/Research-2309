# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# function_code --------------------

def translate_text_to_german(src_text):
    """
    Translates the provided English text to German using the MBart-large-50 model.

    Args:
    src_text (str): The English text to be translated.

    Returns:
    str: Translated text in German.
    """
    # Load the tokenizer and model from the transformers library
    tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50', src_lang='en_XX', tgt_lang='de_DE')
    model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50')

    # Encode the source text
    encoded_text = tokenizer(src_text, return_tensors='pt')
    # Translate the text
    translated_tokens = model.generate(**encoded_text)
    # Decode the translated tokens
    tgt_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    return tgt_text

# test_function_code --------------------

def test_translate_text_to_german():
    print("Testing translate_text_to_german function.")
    # Example English text
    sample_text = 'This is a test sentence for translation purpose.'

    # Expected German translation
    expected_translation = 'Dies ist ein Testsatz für Übersetzungszwecke.' # Note: This is a mock expected translation and may not reflect the actual model output.

    # Testing the translation function
    print("Testing single sentence translation...")
    translated_text = translate_text_to_german(sample_text)
    assert translated_text == expected_translation, f"Translation failed: {translated_text} != {expected_translation}"

    print("Testing finished successfully.")

# Run test function
test_translate_text_to_german()