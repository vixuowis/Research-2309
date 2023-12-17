# requirements_file --------------------

!pip install -U sentence-transformers sklearn

# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def compare_sentence_similarity(sentence1, sentence2):
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
    embeddings = model.encode([sentence1, sentence2])
    similarity_score = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]
    return similarity_score

# test_function_code --------------------

def test_compare_sentence_similarity():
    print("Testing compare_sentence_similarity function...")
    # Test case 1: Similar sentences
    similarity = compare_sentence_similarity("This is a test sentence.", "This is a test sentence.")
    assert similarity > 0.9, f"Similar sentences have low similarity score: {similarity}"

    # Test case 2: Slightly different sentences
    similarity = compare_sentence_similarity("This is a test sentence.", "This is another test sentence.")
    assert 0.7 <= similarity <= 0.9, f"Slightly different sentences have unexpected similarity score: {similarity}"

    # Test case 3: Different sentences
    similarity = compare_sentence_similarity("This is the first sentence.", "Completely different content.")
    assert similarity < 0.5, f"Different sentences have high similarity score: {similarity}"
    print("All tests passed!")

test_compare_sentence_similarity()