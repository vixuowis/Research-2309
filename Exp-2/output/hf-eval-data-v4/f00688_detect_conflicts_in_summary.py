# requirements_file --------------------

!pip install -U sentence_transformers transformers

# function_import --------------------

from sentence_transformers import CrossEncoder

# function_code --------------------

def detect_conflicts_in_summary(summary_sentences):
    """
    Detect if a given summary contains conflicting information.

    Parameters:
    summary_sentences: A list of tuples, with each tuple containing two statements
                       from the summary to be compared.

    Returns:
    A list of dictionaries containing the sentence pairs and their associated
    scores indicating contradiction, entailment, or neutrality.
    """
    model = CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')
    scores = model.predict(summary_sentences)
    results = []
    for pair, score in zip(summary_sentences, scores):
        results.append({'pair': pair, 'score': score})
    return results

# test_function_code --------------------

def test_detect_conflicts_in_summary():
    print("Testing started.")
    test_sentences = [
        ('A man is eating pizza', 'The man is on a diet and avoids eating pizza.'),
        ('A black race car starts up in front of a crowd of people.', 'A man is driving down a lonely road.'),
        ('A child is riding a bike', 'The child hates riding the bike')
    ]
    expected_labels = ['contradiction', 'neutral', 'contradiction']

    results = detect_conflicts_in_summary(test_sentences)

    for i, result in enumerate(results):
        label = 'contradiction' if result['score'][0] > result['score'][1] and result['score'][0] > result['score'][2] else ('entailment' if result['score'][1] > result['score'][2] else 'neutral')
        assert label == expected_labels[i], f"Test case [{i+1}/3] failed: Expected {expected_labels[i]}, got {label}"

    print("Testing finished.")

# Run the test function
test_detect_conflicts_in_summary()