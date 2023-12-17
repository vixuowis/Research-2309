# requirements_file --------------------

!pip install -U transformers sentencepiece

# function_import --------------------

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# function_code --------------------

def translate_welcome_message(english_text: str) -> str:
    # Initialize the model and tokenizer
    model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
    tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M')

    # Set the source language code
    tokenizer.src_lang = 'en'

    # Encode the English text
    encoded_input = tokenizer(english_text, return_tensors='pt')

    # Generate translation in the target language (French)
    generated_tokens = model.generate(**encoded_input, forced_bos_token_id=tokenizer.get_lang_id('fr'))

    # Decode the generated tokens to get the French translation
    french_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return french_text

# test_function_code --------------------

def test_translate_welcome_message():
    print("Testing the translation of welcome message to French.")
    english_text = "Welcome to our hotel, we hope you enjoy your stay."
    expected_output = "Bienvenue à notre hôtel, nous espérons que vous apprécierez votre séjour."
    translated_text = translate_welcome_message(english_text)
    assert translated_text == expected_output, f"Test failed: got '{{translated_text}}'", expected '{{expected_output}}'."
    print("Test passed.")

# Run the test function
test_translate_welcome_message()