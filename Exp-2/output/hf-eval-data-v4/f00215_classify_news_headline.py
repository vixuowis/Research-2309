# requirements_file --------------------

!pip install -U transformers sentence_transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_news_headline(headline):
    classifier = pipeline('zero-shot-classification', model='cross-encoder/nli-deberta-v3-xsmall')
    candidate_labels = ['technology', 'sports', 'politics']
    result = classifier(headline, candidate_labels)
    return result

# test_function_code --------------------

def test_classify_news_headline():
    print("Testing started.")
    # Test case 1: Technology headline
    tech_headline = "Apple just announced the newest iPhone X"
    result = classify_news_headline(tech_headline)
    assert result['labels'][0] == 'technology', f"Test case 1 failed: {result}"

    # Test case 2: Sports headline
    sports_headline = "The Lakers secured their 17th NBA Championship title."
    result = classify_news_headline(sports_headline)
    assert result['labels'][0] == 'sports', f"Test case 2 failed: {result}"

    # Test case 3: Politics headline
    politics_headline = "New legislation has been introduced in Parliament today."
    result = classify_news_headline(politics_headline)
    assert result['labels'][0] == 'politics', f"Test case 3 failed: {result}"
    print("Testing finished.")

test_classify_news_headline()