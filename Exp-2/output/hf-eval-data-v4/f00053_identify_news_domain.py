# requirements_file --------------------

!pip install -U transformers sentence_transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def identify_news_domain(text):
    """
    Identify the domain of the news text based on zero-shot classification.

    Parameters:
    - text (str): The news text to classify.

    Returns:
    - dict: The classification result with probabilities.
    """
    candidate_labels = ['technology', 'sports', 'politics']
    classifier = pipeline('zero-shot-classification', model='cross-encoder/nli-roberta-base')
    result = classifier(text, candidate_labels)
    return result

# test_function_code --------------------

def test_identify_news_domain():
    print("Testing identify_news_domain function.")
    # Test case 1: News about technology.
    tech_news = "Apple just announced the newest iPhone X."
    result = identify_news_domain(tech_news)
    assert result['labels'][0] == 'technology', f"Test case 1 failed: {result}"

    # Test case 2: News about sports.
    sports_news = "The Olympics games are going to be held next year."
    result = identify_news_domain(sports_news)
    assert result['labels'][0] == 'sports', f"Test case 2 failed: {result}"

    # Test case 3: News about politics.
    political_news = "The senate passed the new healthcare bill."
    result = identify_news_domain(political_news)
    assert result['labels'][0] == 'politics', f"Test case 3 failed: {result}"

    print("All test cases passed!")

# Run the test function
test_identify_news_domain()