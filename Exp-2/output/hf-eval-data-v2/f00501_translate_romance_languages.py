# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_romance_languages(romance_languages_documents):
    """
    Translates a list of documents from Romance languages to English.

    Args:
        romance_languages_documents (list): A list of strings where each string is a document in a Romance language.

    Returns:
        list: A list of translated documents in English.

    Raises:
        ValueError: If the input is not a list or if any of the elements in the list is not a string.
    """
    if not isinstance(romance_languages_documents, list) or not all(isinstance(doc, str) for doc in romance_languages_documents):
        raise ValueError('Input should be a list of strings.')

    translate_model = pipeline('translation', model='Helsinki-NLP/opus-mt-ROMANCE-en')
    translated_texts = [translate_model(document)[0]['translation_text'] for document in romance_languages_documents]
    return translated_texts

# test_function_code --------------------

def test_translate_romance_languages():
    """
    Tests the function translate_romance_languages.
    """
    # Test with a list of documents in French, Spanish, and Italian
    documents = ['Je suis un étudiant.', 'Soy un estudiante.', 'Sono uno studente.']
    translations = translate_romance_languages(documents)
    # Check that the output is a list
    assert isinstance(translations, list)
    # Check that the length of the output list is the same as the input list
    assert len(translations) == len(documents)
    # Check that all elements in the output list are strings
    assert all(isinstance(translation, str) for translation in translations)

# call_test_function_code --------------------

test_translate_romance_languages()