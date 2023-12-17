# requirements_file --------------------

import subprocess

requirements = ["sentence_transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from sentence_transformers import CrossEncoder

# function_code --------------------

def classify_news_article(news_article, candidate_labels):
    """Classify the content of a news article into given categories without labeled data.

    Args:
        news_article (str): The content of the news article to classify.
        candidate_labels (list): A list of category labels to classify the article into.

    Returns:
        dict: A dictionary with labels as keys and their corresponding scores as values.

    Raises:
        ValueError: If news_article is empty or candidate_labels is empty or not a list.
    """
    if not news_article:
        raise ValueError('news_article must not be empty')
    if not candidate_labels or not isinstance(candidate_labels, list):
        raise ValueError('candidate_labels must be a non-empty list')

    cross_encoder = CrossEncoder('cross-encoder/nli-roberta-base')
    scores = cross_encoder.predict([
        {'sentence1': news_article, 'sentence2': label} for label in candidate_labels
    ])
    return dict(zip(candidate_labels, scores))

# test_function_code --------------------

def test_classify_news_article():
    print("Testing started.")
    news_articles = [
        'Apple just released the newest iPhone X which introduces cutting edge technology.',
        'Last night, the local basketball team won their game with a thrilling finish.',
        'Recent elections have changed the political landscape of the country.',
    ]
    candidate_labels = ['technology', 'sports', 'politics']
    
    # Testing case 1
    print("Testing case [1/3] started.")
    res = classify_news_article(news_articles[0], candidate_labels)
    assert 'technology' in res, f"Test case [1/3] failed: Expected 'technology', got {res}"
    
    # Testing case 2
    print("Testing case [2/3] started.")
    res = classify_news_article(news_articles[1], candidate_labels)
    assert 'sports' in res, f"Test case [2/3] failed: Expected 'sports', got {res}"
    
    # Testing case 3
    print("Testing case [3/3] started.")
    res = classify_news_article(news_articles[2], candidate_labels)
    assert 'politics' in res, f"Test case [3/3] failed: Expected 'politics', got {res}"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_news_article()