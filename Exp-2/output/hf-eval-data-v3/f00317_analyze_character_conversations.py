# function_import --------------------

from sentence_transformers import SentenceTransformer
import numpy as np

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
    similarity_score = np.mean([np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) for a, b in zip(embeddings_A, embeddings_B)])
    return similarity_score

# test_function_code --------------------

def test_analyze_character_conversations():
    """
    Test the function analyze_character_conversations.
    """
    conversation_A = ["I think we should go there.", "What do you recommend?"]
    conversation_B = ["Let's check that place.", "Which one do you suggest?"]
    similarity_score = analyze_character_conversations(conversation_A, conversation_B)
    assert 0 <= similarity_score <= 1, 'Similarity score should be between 0 and 1'
    
    conversation_A = ["I love apples.", "I hate oranges."]
    conversation_B = ["I love oranges.", "I hate apples."]
    similarity_score = analyze_character_conversations(conversation_A, conversation_B)
    assert 0 <= similarity_score <= 1, 'Similarity score should be between 0 and 1'
    
    conversation_A = ["The weather is nice today.", "I love sunny days."]
    conversation_B = ["The weather is terrible today.", "I hate rainy days."]
    similarity_score = analyze_character_conversations(conversation_A, conversation_B)
    assert 0 <= similarity_score <= 1, 'Similarity score should be between 0 and 1'
    
    return 'All Tests Passed'

# call_test_function_code --------------------

test_analyze_character_conversations()