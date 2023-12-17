# requirements_file --------------------

!pip install -U 

# function_import --------------------

from sentence_transformers import CrossEncoder

# function_code --------------------

def analyze_text_relationship(sentence1, sentence2):
    """
    Analyze the logical relationship between two sentences using a pre-trained model.

    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.

    Returns:
        dict: A dictionary containing the scores for 'contradiction', 'entailment', and 'neutral'.
    """
    model = CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')
    scores = model.predict([(sentence1, sentence2)])
    return {'contradiction': scores[0][0], 'entailment': scores[0][1], 'neutral': scores[0][2]}

# test_function_code --------------------

def test_analyze_text_relationship():
    print("Testing started.")

    # Test case 1: Entailment
    print("Testing case [1/3] started.")
    scores = analyze_text_relationship('A man is eating pizza', 'A man eats something')
    assert max(scores, key=scores.get) == 'entailment', f"Test case [1/3] failed: Expected 'entailment', got {max(scores, key=scores.get)}"

    # Test case 2: Contradiction
    print("Testing case [2/3] started.")
    scores = analyze_text_relationship('A bird is flying', 'No bird is in the sky')
    assert max(scores, key=scores.get) == 'contradiction', f"Test case [2/3] failed: Expected 'contradiction', got {max(scores, key=scores.get)}"

    # Test case 3: Neutral
    print("Testing case [3/3] started.")
    scores = analyze_text_relationship('A dog runs in the park', 'There is a park in the city')
    assert max(scores, key=scores.get) == 'neutral', f"Test case [3/3] failed: Expected 'neutral', got {max(scores, key=scores.get)}"
    print("Testing finished.")

# Run the test function
test_analyze_text_relationship()