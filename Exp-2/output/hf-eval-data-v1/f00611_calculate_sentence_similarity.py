from sentence_transformers import SentenceTransformer
import numpy as np

# Function to calculate sentence similarity
# This function uses the Hugging Face Transformers library and the SentenceTransformer class
# to load a pre-trained model and calculate the similarity between two sentences.
# The model used is 'flax-sentence-embeddings/all_datasets_v4_MiniLM-L6', which has been trained for sentence similarity tasks.
# The function takes two sentences as input, processes and encodes them using the model, and then calculates the similarity between the two sentence embeddings.
def calculate_sentence_similarity(sentence1, sentence2):
    # Load the pre-trained model
    model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
    # Process and encode the sentences
    embedding1 = model.encode(sentence1)
    embedding2 = model.encode(sentence2)
    # Calculate the similarity between the two sentence embeddings
    similarity = np.inner(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity