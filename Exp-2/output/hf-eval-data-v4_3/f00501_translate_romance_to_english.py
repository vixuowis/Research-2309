# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_romance_to_english(documents):
    """
    Translates a list of documents from Romance languages to English.

    Args:
        documents (List[str]): A list of text documents in Romance languages.

    Returns:
        List[str]: A list of translated documents in English.

    Raises:
        Exception: If the translation model fails to load or translation fails.
    """
    try:
        translate_model = pipeline('translation', model='Helsinki-NLP/opus-mt-ROMANCE-en')
        return [translate_model(document) for document in documents]
    except Exception as e:
        raise Exception('Translation failed: ' + str(e))

# test_function_code --------------------

def test_translate_romance_to_english():
    print('Testing started.')
    sample_documents = [
        'Bonjour, comment ça va ?',  # French
        'Árbol significa ‘tree’ en inglés.',  # Spanish
        'Buona giornata a tutti!'  # Italian
    ]

    print('Testing case [1/3] started.')
    try:
        translated_docs = translate_romance_to_english(sample_documents)
        assert len(translated_docs) == 3 and all(isinstance(doc, str) for doc in translated_docs), 'Test case [1/3] failed: Incorrect translation output type or length.'
    except Exception as e:
        assert False, f'Test case [1/3] failed with exception: {e}'

    print('Testing case [2/3] started.')
    assert translated_docs[0].startswith('Hello'), 'Test case [2/3] failed: French to English translation error.'
    print('Testing case [3/3] started.')
    assert translated_docs[2].startswith('Good day'), 'Test case [3/3] failed: Italian to English translation error.'
    print('Testing finished.')

# call_test_function_line --------------------

test_translate_romance_to_english()