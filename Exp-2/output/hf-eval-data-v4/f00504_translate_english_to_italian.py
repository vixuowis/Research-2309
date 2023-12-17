# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_english_to_italian(text):
    """
    Translate English text to Italian using Hugging Face's translation model.

    Parameters:
        text (str): The English text to be translated.

    Returns:
        str: The translated Italian text.
    """
    # Initialize the translation model
    translator = pipeline('translation_en_to_it', model='Helsinki-NLP/opus-mt-en-it')

    # Perform the translation
    translation = translator(text, max_length=512)

    # Extract the translated text
    return translation[0]['translation_text']

# test_function_code --------------------

def test_translate_english_to_italian():
    print("Testing started.")
    
    # Test case 1: Basic greeting
    print("Testing case [1/3] started.")
    assert translate_english_to_italian('Hello, world!') == 'Ciao, mondo!', "Test case [1/3] failed: Basic greeting translation mismatch."

    # Test case 2: Longer sentence
    print("Testing case [2/3] started.")
    assert translate_english_to_italian('Welcome to our website. Discover our products and services.') == 'Benvenuti sul nostro sito web. Scopri i nostri prodotti e servizi.', "Test case [2/3] failed: Longer sentence translation mismatch."

    # Test case 3: Handling special characters
    print("Testing case [3/3] started.")
    assert translate_english_to_italian("Let's try translating this sentence, shall we?") == "Proviamo a tradurre questa frase, d'accordo?", "Test case [3/3] failed: Special characters handling mismatch."
    
    print("Testing finished.")