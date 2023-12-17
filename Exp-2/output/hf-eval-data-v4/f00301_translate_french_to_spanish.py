# requirements_file --------------------

!pip install -U torch tokenizers transformers

# function_import --------------------

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# function_code --------------------

def translate_french_to_spanish(input_text):
    """
    Translate a given text from French to Spanish using a pre-trained model.

    Parameters:
    input_text (str): The French text to be translated.

    Returns:
    str: The translated Spanish text.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-fr-es')
    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-es')
    tokenized_input = tokenizer(input_text, return_tensors='pt')
    output_tokens = model.generate(**tokenized_input)
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return output_text

# test_function_code --------------------

def test_translate_french_to_spanish():
    print("Testing translation function.")
    
    # Test case 1: Simple greeting
    french_text = "Bonjour, comment ça va?"
    expected_spanish = "Hola, ¿cómo estás?"  # Expected output may vary
    translated_text = translate_french_to_spanish(french_text)
    assert translated_text == expected_spanish, f"Test case failed: expected {expected_spanish}, got {translated_text}"
    
    print("All test cases passed!")

# Running the test function
test_translate_french_to_spanish()