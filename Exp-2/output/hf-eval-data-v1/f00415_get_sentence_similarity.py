from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to get sentence similarity
# This function uses the SentenceTransformer from the Hugging Face Transformers library
# to encode sentences into embeddings and then compares these embeddings using cosine similarity
# to find similar sentences.
def get_sentence_similarity(sentences):
    # Load the pretrained model
    model = SentenceTransformer('nikcheerla/nooks-amd-detection-v2-full')
    # Encode the sentences to get their embeddings
    embeddings = model.encode(sentences)
    # Calculate the cosine similarity between the sentence embeddings
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix