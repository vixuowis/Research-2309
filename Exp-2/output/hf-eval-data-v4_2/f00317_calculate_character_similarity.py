# requirements_file --------------------

pip install -U sentence-transformers numpy

# function_import --------------------

from sentence_transformers import SentenceTransformer

# function_code --------------------

def calculate_character_similarity(conversations):
    """
    Calculate the similarity between conversations of different characters.

    Args:
        conversations (dict): A dictionary where keys are character names and values are lists of sentences (conversations).

    Returns:
        dict: A dictionary with key pairs as tuples of character names and values as similarity scores.

    Raises:
        ValueError: If the input is not a dictionary or lists of conversations are empty.
    """
    # Ensure input is a dictionary and conversations are not empty
    if not isinstance(conversations, dict) or any(len(convs) == 0 for convs in conversations.values()):
        raise ValueError('Input must be a dictionary with non-empty lists of conversations.')

    model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')

    # Encode conversations
    char_embeddings = {name: model.encode(convos) for name, convos in conversations.items()}

    # Calculate similarity scores
    similarity_scores = {}
    for name1, emb1 in char_embeddings.items():
        for name2, emb2 in char_embeddings.items():
            if name1 < name2: # Prevent duplicate pairs
                score = np.mean([np.dot(e1, e2) for e1, e2 in zip(emb1, emb2)])
                similarity_scores[(name1, name2)] = score
    return similarity_scores

# test_function_code --------------------

def test_calculate_character_similarity():
    print("Testing started.")
    # Sample conversations data
    sample_conversations = {
        'Alice': ['The weather is nice today.', 'Shall we go for a walk?'],
        'Bob': ['Today is a good day to stay inside.', 'Maybe we can read a book.'],
        'Charlie': ['I want to play outside.', 'It looks like it might rain though.']
    }

    # Testing case 1: Correct input
    print("Testing case [1/2] started.")
    similarities = calculate_character_similarity(sample_conversations)
    assert isinstance(similarities, dict), "Test case [1/2] failed: The result should be a dictionary."

    # Testing case 2: Incorrect input
    print("Testing case [2/2] started.")
    try:
        calculate_character_similarity([])
        assert False, "Test case [2/2] failed: ValueError not raised with incorrect input."
    except ValueError:
        pass
    print("Testing finished.")

# call_test_function_line --------------------

test_calculate_character_similarity()