# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def find_most_related_faq(faq_sentences: list, query: str) -> str:
    """
    Find the most related FAQ for a given customer query using SentenceTransformer.

    Args:
        faq_sentences (list): A list of FAQ sentences.
        query (str): A customer query.

    Returns:
        str: The most related FAQ for the given customer query.
    """

    transformer = SentenceTransformer("bert-base-nli-mean-tokens") # 768 dims, binary tokenization
    
    corpus_embeddings = transformer.encode(faq_sentences) # (1024, n)
    query_embedding = transformer.encode([query]) # (1024, 1)
    
    cosine_similarities = cosine_similarity(query_embedding, corpus_embeddings).flatten() # (n,)
    best_faq_index = cosine_similarities.argmax()
    
    return faq_sentences[best_faq_index]

# test_function_code --------------------

def test_find_most_related_faq():
    """Test the function find_most_related_faq."""
    faq_sentences = ["FAQ1 text", "FAQ2 text", "FAQ3 text"]
    query = "Customer query"
    assert isinstance(find_most_related_faq(faq_sentences, query), str)
    faq_sentences = ["What is your name?", "How old are you?", "Where are you from?"]
    query = "What's your age?"
    assert find_most_related_faq(faq_sentences, query) == "How old are you?"
    return 'All Tests Passed'


# call_test_function_code --------------------

test_find_most_related_faq()