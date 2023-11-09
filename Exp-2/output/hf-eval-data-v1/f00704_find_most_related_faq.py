from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def find_most_related_faq(faq_sentences, query):
    '''
    This function finds the most related FAQ for a given customer query using SentenceTransformer.
    Args:
    faq_sentences (list): A list of FAQ sentences.
    query (str): A customer query.
    Returns:
    str: The most related FAQ.
    '''
    # Create an instance of the SentenceTransformer class with the pre-trained model
    model = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2')
    # Encode the customer query and the FAQ sentences
    embeddings = model.encode(faq_sentences + [query])
    # Get the embedding of the customer query
    query_embedding = embeddings[-1]
    # Calculate the cosine similarity between the encoded vectors of the customer query and the FAQ sentences
    sim_scores = cosine_similarity([query_embedding], embeddings[:-1])
    # Get the index of the FAQ with the highest cosine similarity score
    most_related_faq_index = sim_scores.argmax()
    # Get the most related FAQ
    most_related_faq = faq_sentences[most_related_faq_index]
    return most_related_faq