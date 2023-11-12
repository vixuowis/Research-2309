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
    model = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2')
    embeddings = model.encode(faq_sentences + [query])
    query_embedding = embeddings[-1]
    sim_scores = cosine_similarity([query_embedding], embeddings[:-1])
    most_related_faq_index = sim_scores.argmax()
    return faq_sentences[most_related_faq_index]

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