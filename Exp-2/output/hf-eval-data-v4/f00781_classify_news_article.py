# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_news_article(news_article):
    classifier = pipeline('zero-shot-classification', model='typeform/squeezebert-mnli')
    candidate_labels = ['Politics', 'Sports', 'Technology', 'Business', 'Entertainment']
    return classifier(news_article, candidate_labels)

# test_function_code --------------------

def test_classify_news_article():
    print("Testing started.")
    example_text = "The government passed a new law today that affects the tech industry."

    # Test case 1
    print("Testing case [1/3] started.")
    result = classify_news_article(example_text)
    assert 'labels' in result and 'scores' in result, "Test case [1/3] failed: 'labels' or 'scores' not in the result."
    assert type(result['labels']) is list, "Test case [1/3] failed: The 'labels' field is not a list."
    assert type(result['scores']) is list, "Test case [1/3] failed: The 'scores' field is not a list."
    print("Testing case [1/3] passed.")

    print("Testing finished.")

    # Run the test function
test_classify_news_article()