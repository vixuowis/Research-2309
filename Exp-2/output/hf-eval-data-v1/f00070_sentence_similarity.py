from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Function to calculate the similarity between two sentences
# using the Hugging Face Transformers library
# The function uses the 'sentence-transformers/paraphrase-MiniLM-L3-v2' model
# to encode the sentences into dense vector representations
# The similarity between the sentences is then calculated using cosine similarity

def sentence_similarity(sentence1: str, sentence2: str) -> float:
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
    embeddings = model.encode([sentence1, sentence2])
    similarity_score = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))
    return similarity_score[0][0]