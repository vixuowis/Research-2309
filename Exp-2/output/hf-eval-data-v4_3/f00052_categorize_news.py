# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def categorize_news(article):
    """Categorize a French news article into sports, politics, or science.

    Args:
        article (str): The content of the news article to be categorized.

    Returns:
        dict: A dictionary with categories as keys and their respective probabilities as values.

    Raises:
        ValueError: If the article content is empty or None.
    """
    if not article:
        raise ValueError('The article content must not be empty.')
    classifier = pipeline('zero-shot-classification', model='BaptisteDoyen/camembert-base-xnli')
    candidate_labels = ['sports', 'politics', 'science']
    hypothesis_template = 'Ce texte parle de {}.'
    result = classifier(article, candidate_labels, hypothesis_template=hypothesis_template)
    return result

# test_function_code --------------------

def test_categorize_news():
    print('Testing started.')
    # Testing cases
    article_sports = 'L\u00e9quipe de France joue aujourd\u2019hui au Parc des Princes.'
    article_politics = 'La nouvelle loi de r\u00e9forme fiscale a \u00e9t\u00e9 annonc\u00e9e aujourd\u2019hui.'
    article_science = 'La recherche sur les cellules souches a fait une perc\u00e9e importante.'

    # Test case 1: Categorize a sports article
    print('Testing case [1/3] started.')
    result_sports = categorize_news(article_sports)
    assert 'sports' in result_sports['labels'], 'Test case [1/3] failed: Sports category not detected.'

    # Test case 2: Categorize a politics article
    print('Testing case [2/3] started.')
    result_politics = categorize_news(article_politics)
    assert 'politics' in result_politics['labels'], 'Test case [2/3] failed: Politics category not detected.'

    # Test case 3: Categorize a science article
    print('Testing case [3/3] started.')
    result_science = categorize_news(article_science)
    assert 'science' in result_science['labels'], 'Test case [3/3] failed: Science category not detected.'
    print('Testing finished.')

# call_test_function_line --------------------

test_categorize_news()