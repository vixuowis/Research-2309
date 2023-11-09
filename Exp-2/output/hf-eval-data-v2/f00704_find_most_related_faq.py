# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def find_most_related_faq(faq_sentences, query):
    """
    This function finds the most related FAQ for a given customer query using SentenceTransformer.

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
    """
    This function tests the find_most_related_faq function.
    """
    faq_sentences = ['What is your return policy?', 'What is your shipping policy?', 'Do you offer discounts?']
    query = 'Can I return my purchase?'
    assert find_most_related_faq(faq_sentences, query) in faq_sentences

# call_test_function_code --------------------

test_find_most_related_faq()