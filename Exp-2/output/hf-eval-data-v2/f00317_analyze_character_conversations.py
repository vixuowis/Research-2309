# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def analyze_character_conversations(conversation_A, conversation_B):
    """
    Analyze how characters in a book are connected and if they share any similarity based on their conversation.

    Args:
        conversation_A (list): List of sentences spoken by character A.
        conversation_B (list): List of sentences spoken by character B.

    Returns:
        float: Similarity score between the conversations of the two characters.
    """
    model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')

    embeddings_A = model.encode(conversation_A)
    embeddings_B = model.encode(conversation_B)

    # Compute similarity between conversations
    similarity = cosine_similarity(embeddings_A.mean(axis=0).reshape(1, -1), embeddings_B.mean(axis=0).reshape(1, -1))[0][0]

    return similarity

# test_function_code --------------------

def test_analyze_character_conversations():
    """
    Test the function analyze_character_conversations.
    """
    conversation_A = ["I think we should go there.", "What do you recommend?"]
    conversation_B = ["Let's check that place.", "Which one do you suggest?"]
    conversation_C = ["The weather is nice today.", "I love ice cream."]

    assert 0.7 <= analyze_character_conversations(conversation_A, conversation_B) <= 1.0
    assert 0.0 <= analyze_character_conversations(conversation_A, conversation_C) <= 0.3

# call_test_function_code --------------------

test_analyze_character_conversations()