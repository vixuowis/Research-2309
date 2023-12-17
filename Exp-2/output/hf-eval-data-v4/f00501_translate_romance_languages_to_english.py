# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_romance_languages_to_english(documents):
    """
    Translate a list of documents from Romance languages to English.
    
    Parameters:
        documents (list): A list of strings, where each string is a document in a Romance language.
    
    Returns:
        list: A list of translated documents in English.
    """
    translate_model = pipeline('translation', model='Helsinki-NLP/opus-mt-ROMANCE-en')
    return [translate_model(document)[0]['translation_text'] for document in documents]

# test_function_code --------------------

def test_translate_romance_languages_to_english():
    print("Testing translate_romance_languages_to_english function.")
    # Example documents in Romance languages
    romance_documents = [
        'Bonjour, comment ça va?',  # French
        'Hola, ¿qué tal?',            # Spanish
        'Ciao, come stai?'             # Italian
    ]
    # Expected translations
    expected_translations = [
        'Hello, how are you?',
        'Hello, how are you?',
        'Hello, how are you?'
    ]
    # Test the function
    translations = translate_romance_languages_to_english(romance_documents)
    assert translations == expected_translations, "Test failed: translations do not match expected values"
    print("All tests passed successfully.")

# Run the test
test_translate_romance_languages_to_english()