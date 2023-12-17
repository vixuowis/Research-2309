# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def categorize_french_news_articles(text):
    classifier = pipeline('zero-shot-classification', model='BaptisteDoyen/camembert-base-xnli')
    candidate_labels = ['sport', 'politique', 'science']
    hypothesis_template = "Ce texte parle de {}."
    return classifier(text, candidate_labels, hypothesis_template=hypothesis_template)

# test_function_code --------------------

def test_categorize_french_news_articles():
    print("Testing started.")
    test_cases = [
        ("L'quipe de France joue aujourd'hui au Parc des Princes", "sport"),
        ("Le prsident annonce de nouvelles lois", "politique"),
        ("La dernire dcouverte scientifique", "science")
    ]

    for i, (text, expected_label) in enumerate(test_cases, start=1):
        print(f"Testing case [{i}/" + str(len(test_cases)) + "] started.")
        result = categorize_french_news_articles(text)
        matched_label = max(result['labels'], key=lambda label: result['scores'][result['labels'].index(label)])
        assert matched_label.lower() == expected_label.lower(), f"Test case [{i}/" + str(len(test_cases)) + "] failed: expected '" + expected_label + "' but got '" + matched_label + "'"
        print(f"Testing case [{i}/" + str(len(test_cases)) + "] passed.")
    print("Testing finished.")

test_categorize_french_news_articles()