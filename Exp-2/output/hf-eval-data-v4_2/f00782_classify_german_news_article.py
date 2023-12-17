# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_german_news_article(article_text):
    """
    Classify a German news article into predefined categories using zero-shot classification.

    Args:
        article_text (str): The German news article text to classify.

    Returns:
        dict: The classification results including category and confidence scores.

    Raises:
        ValueError: If the article_text is not a string.
    """
    if not isinstance(article_text, str):
        raise ValueError('The article_text must be a string.')

    classifier = pipeline('zero-shot-classification', model='Sahajtomar/German_Zeroshot')
    candidate_labels = ['Verbrechen', 'Tragödie', 'Stehlen']
    hypothesis_template = 'In diesem Text geht es um {}.'
    return classifier(article_text, candidate_labels, hypothesis_template=hypothesis_template)

# test_function_code --------------------

def test_classify_german_news_article():
    print("Testing started.")
    # Test cases using example German news articles

    # Testing case: Should classify as 'Tragödie'
    print("Testing case [1/3] started.")
    article_text = 'Letzte Woche gab es einen Selbstmord in einer nahe gelegenen Kolonie.'
    result = classify_german_news_article(article_text)
    assert result['labels'][0] == 'Tragödie', f"Test case [1/3] failed: {result}"

    # Testing case: Should classify as 'Verbrechen'
    print("Testing case [2/3] started.")
    article_text = 'Ein Bankraub wurde gestern in der Innenstadt verügt.'
    result = classify_german_news_article(article_text)
    assert result['labels'][0] == 'Verbrechen', f"Test case [2/3] failed: {result}"

    # Testing case: Should classify as 'Stehlen'
    print("Testing case [3/3] started.")
    article_text = 'Ein Diebstahl wurde in einem lokalen Geschäft gemeldet.'
    result = classify_german_news_article(article_text)
    assert result['labels'][0] == 'Stehlen', f"Test case [3/3] failed: {result}"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_german_news_article()