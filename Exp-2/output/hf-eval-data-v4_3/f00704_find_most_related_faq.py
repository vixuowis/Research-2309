# requirements_file --------------------

import subprocess

requirements = ["sentence-transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def find_most_related_faq(faq_sentences, query):
    """
    Find the most related FAQ for a given customer query using sentence-transformers model.

    Args:
        faq_sentences (list): A list of FAQ sentences to be compared.
        query (str): The customer query.

    Returns:
        str: The most related FAQ sentence.

    Raises:
        ValueError: If the faq_sentences list is empty or the query is not provided.
    """
    if not faq_sentences:
        raise ValueError('The faq_sentences list cannot be empty.')
    if not query:
        raise ValueError('The query cannot be empty.')
    
    model = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2')
    embeddings = model.encode(faq_sentences + [query])
    query_embedding = embeddings[-1]
    sim_scores = cosine_similarity([query_embedding], embeddings[:-1])
    most_related_faq_index = sim_scores.argmax()
    return faq_sentences[most_related_faq_index]

# test_function_code --------------------

def test_find_most_related_faq():
    print("Testing started.")
    faq_sentences = ['What is your return policy?', 'How can I track my order?', 'Do you offer international shipping?']
    query = 'How can I return a product?'
    
    # Testing case 1
    print("Testing case [1/1] started.")
    most_related_faq = find_most_related_faq(faq_sentences, query)
    assert most_related_faq == 'What is your return policy?', f"Test case [1/1] failed: Expected 'What is your return policy?', but got {most_related_faq}"
    print("Testing finished.")

# call_test_function_line --------------------

test_find_most_related_faq()