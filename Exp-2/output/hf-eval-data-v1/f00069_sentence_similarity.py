from sentence_transformers import SentenceTransformer
import numpy as np

# This function is used to compare and contrast user input sentences with existing sentences in our database.
# It uses the Hugging Face Transformers SentenceTransformer to compute the embeddings for a set of sentences.
# The model used is 'sentence-transformers/paraphrase-distilroberta-base-v2', which maps sentences & paragraphs to a 768 dimensional dense vector space.
# These embeddings can be used for tasks like clustering or semantic search.
def sentence_similarity(user_input_sentences):
    # Create a SentenceTransformer object
    model = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v2')
    # Compute the embeddings for the user input sentences
    embeddings = model.encode(user_input_sentences)
    return embeddings