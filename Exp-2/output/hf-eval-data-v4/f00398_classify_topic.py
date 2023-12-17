# requirements_file --------------------

!pip install -U sentence_transformers t=3.5.1 transformers =4.18.0

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_topic(sentence):
    """
    Classify the given sentence into categories: 'technology', 'literature', 'science'.

    Args:
    sentence (str): The sentence to be classified.

    Returns:
    dict: The classification scores for each category.
    """
    classifier = pipeline('zero-shot-classification', model='cross-encoder/nli-deberta-v3-xsmall')
    candidate_labels = ['technology', 'literature', 'science']
    result = classifier(sentence, candidate_labels)
    return result

# test_function_code --------------------

def test_classify_topic():
    print("Testing classify_topic function.")
    # Test case 1: Sentence related to technology
    sentence1 = 'Quantum computing will revolutionize information processing.'
    result1 = classify_topic(sentence1)
    assert 'technology' in result1['labels'], f"Test case 1 failed: {result1}"

    # Test case 2: Sentence related to literature
    sentence2 = 'Shakespeare's works are pivotal in English literature.'
    result2 = classify_topic(sentence2)
    assert 'literature' in result2['labels'], f"Test case 2 failed: {result2}"

    # Test case 3: Sentence related to science
    sentence3 = 'The discovery of CRISPR has huge implications for genetics.'
    result3 = classify_topic(sentence3)
    assert 'science' in result3['labels'], f"Test case 3 failed: {result3}"

    print("All test cases passed.")

# Run the test function
test_classify_topic()