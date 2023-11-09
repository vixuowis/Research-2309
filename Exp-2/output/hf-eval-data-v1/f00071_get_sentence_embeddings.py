from sentence_transformers import SentenceTransformer
import numpy as np

# Function to get sentence embeddings
# Input: List of sentences
# Output: List of sentence embeddings

def get_sentence_embeddings(sentences):
    '''
    This function takes a list of sentences and returns their embeddings using the SentenceTransformer model.
    The model is trained to derive embeddings for sentences, which can represent their semantic meaning in a 768-dimensional vector space.
    '''
    # Instantiate a model with the 'sentence-transformers/nli-mpnet-base-v2'
    model = SentenceTransformer('sentence-transformers/nli-mpnet-base-v2')
    # Encode the sentences, transforming them into a dense representation
    encoded_sentences = model.encode(sentences)
    return encoded_sentences