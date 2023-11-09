from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def find_most_similar_sentences(sentences):
    '''
    This function takes a list of sentences as input and returns the pair of sentences that are most similar.
    The similarity is calculated using the SentenceTransformer from the Hugging Face Transformers library.
    '''
    # Create the model
    model = SentenceTransformer('sentence-transformers/distilbert-base-nli-mean-tokens')
    
    # Encode the sentences into dense vector representations
    embeddings = model.encode(sentences)
    
    # Compute the cosine similarity for each pair of sentences
    similarities = cosine_similarity(embeddings)
    
    # Find the indices of the most similar pair of sentences
    indices = np.unravel_index(np.argmax(similarities, axis=None), similarities.shape)
    
    return sentences[indices[0]], sentences[indices[1]]