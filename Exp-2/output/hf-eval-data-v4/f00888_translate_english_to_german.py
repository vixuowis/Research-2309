# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def translate_english_to_german(text):
    # Instantiating the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')

    # Preparing the text for translation
    input_text = f'translate English to German: {text}'
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids

    # Generating the translation
    outputs = model.generate(input_ids)

    # Decoding and returning the translation
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# test_function_code --------------------

def test_translate_english_to_german():
    print("Testing translation function.")

    # Test case 1: Simple sentence
    source_text = "I have a doctor's appointment tomorrow morning."
    expected_translation = "Ich habe morgen früh einen Arzttermin."
    actual_translation = translate_english_to_german(source_text)
    assert actual_translation == expected_translation, f"Test case 1 failed: Expected {{expected_translation}}, got {{actual_translation}}"

    # Test case 2: Ensure correct working with punctuation
    source_text = "The weather is great, isn't it?"
    expected_translation = "Das Wetter ist großartig, nicht wahr?"
    actual_translation = translate_english_to_german(source_text)
    assert actual_translation == expected_translation, f"Test case 2 failed: Expected {{expected_translation}}, got {{actual_translation}}"

    print("Testing completed successfully.")

# Run the test function
test_translate_english_to_german()