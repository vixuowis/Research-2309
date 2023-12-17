# requirements_file --------------------

!pip install -U sentence-transformers sklearn

# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def find_most_related_faq(faq_sentences, query):
    """
    Finds the most related FAQ in a list for a given customer query using semantic similarity.

    Parameters:
        faq_sentences (list): A list of FAQ sentences.
        query (str): A customer query string.

    Returns:
        str: The FAQ sentence that is most semantically similar to the customer query.
    """
    model = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2')
    embeddings = model.encode(faq_sentences + [query])
    query_embedding = embeddings[-1]
    sim_scores = cosine_similarity([query_embedding], embeddings[:-1])
    most_related_faq_index = sim_scores.argmax()
    return faq_sentences[most_related_faq_index]

# test_function_code --------------------

def test_find_most_related_faq():
    print("Testing started.")
    faq_sentences = [
        "How can I reset my password?",
        "What is the refund policy?",
        "How to track my order?"
    ]
    query = "I forgot my password, how do I recover it?"
    most_related_faq = find_most_related_faq(faq_sentences, query)
    assert most_related_faq == faq_sentences[0], f"Test failed: Expected FAQ '{{faq_sentences[0]}}' but got '{{most_related_faq}}' instead."
    print("Testing finished.")
    
# Run the test function
test_find_most_related_faq()