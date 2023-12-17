# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModel, AutoTokenizer

# function_code --------------------

def translate_swedish_to_english(swedish_text):
    """
    Translate Swedish text to English using Hugging Face Transformers.

    :param swedish_text: str - the Swedish text to be translated
    :return: str - the translated English text
    """
    # Load the pre-trained model and tokenizer
    model_name = 'Helsinki-NLP/opus-mt-sv-en'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize the input text and convert to tensor format
    inputs = tokenizer.encode(swedish_text, return_tensors='pt')

    # Translate the text
    translated = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)

    # Convert the generated tokens to text
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    return translated_text

# test_function_code --------------------

def test_translate_swedish_to_english():
    print("Testing translate_swedish_to_english function.")

    # Example Swedish text
    swedish_text = "Stockholm är Sveriges huvudstad och största stad. Den har en rik historia och erbjuder många kulturella och historiska sevärdheter."

    # Expected translation
    expected_translation = "Stockholm is the capital and largest city of Sweden. It has a rich history and offers many cultural and historical attractions."

    # Translate the text
    translation = translate_swedish_to_english(swedish_text)

    # Check the result
    assert translation == expected_translation, f"Failed to translate correctly: {translation}"
    print("Test passed!")

# Run the test function
test_translate_swedish_to_english()