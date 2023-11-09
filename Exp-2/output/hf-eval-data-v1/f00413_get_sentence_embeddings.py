from sentence_transformers import SentenceTransformer
import numpy as np

# Function to get sentence embeddings
# This function takes a list of sentences as input and returns their embeddings
# The embeddings are calculated using the SentenceTransformer model from the Hugging Face Transformers library
# The model used is 'sentence-transformers/bert-base-nli-mean-tokens' which maps sentences to a 768 dimensional vector space

def get_sentence_embeddings(sentences):
    # Initialize the SentenceTransformer model
    model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    # Get the embeddings for the input sentences
    embeddings = model.encode(sentences)
    return embeddings