# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_romance_languages_to_english(romance_languages_documents):
    '''
    Translate texts from multiple Romance languages to English.

    Args:
        romance_languages_documents (list): A list of documents in Romance languages.

    Returns:
        list: A list of translated English texts.

    Raises:
        ValueError: If the input is not a list or if it's empty.
    '''
    if not isinstance(romance_languages_documents, list) or not romance_languages_documents:
        raise ValueError('Input should be a non-empty list of documents.')
    translate_model = pipeline('translation', model='Helsinki-NLP/opus-mt-ROMANCE-en')
    translated_texts = [translate_model(document)[0]['translation_text'] for document in romance_languages_documents]
    return translated_texts

# test_function_code --------------------

def test_translate_romance_languages_to_english():
    '''
    Test the function translate_romance_languages_to_english.
    '''
    # Test with French, Spanish and Italian texts
    documents = ['Je suis un étudiant.', 'Soy un estudiante.', 'Sono uno studente.']
    translations = translate_romance_languages_to_english(documents)
    assert isinstance(translations, list), 'The result should be a list.'
    assert len(translations) == len(documents), 'The number of translations should be equal to the number of input documents.'
    assert all(isinstance(text, str) for text in translations), 'All translations should be strings.'
    # Test with an empty list
    try:
        translate_romance_languages_to_english([])
    except ValueError as e:
        assert str(e) == 'Input should be a non-empty list of documents.', 'The function should raise a ValueError with an appropriate message when the input list is empty.'
    # Test with a non-list input
    try:
        translate_romance_languages_to_english('I am a student.')
    except ValueError as e:
        assert str(e) == 'Input should be a non-empty list of documents.', 'The function should raise a ValueError with an appropriate message when the input is not a list.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_romance_languages_to_english()