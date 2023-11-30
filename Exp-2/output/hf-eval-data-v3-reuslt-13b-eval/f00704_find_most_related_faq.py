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
    
    # Load model from Sentence Transformer. 

    sbert_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')  
    
    # Get vector representations for query and FAQ sentences (FAQ sentences need to be tokenized first).
    
    query_vector = sbert_model.encode([query], show_progress_bar=True)
    faq_vector = sbert_model.encode(faq_sentences, show_progress_bar=True)
    
    # Calculate cosine similarity between FAQ vectors and the query vector.
    
    sim_scores = cosine_similarity(query_vector, faq_vector)[0]
    
    # Get the index of the most related FAQ sentence. 
    
    max_score = np.argmax(sim_scores)
    
    # Return the most related FAQ sentence (original format).
    
    return faq_sentences[max_score]  


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