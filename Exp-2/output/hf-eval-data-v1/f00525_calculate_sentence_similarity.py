from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_sentence_similarity(sentences):
    '''
    This function calculates the similarity between sentences using the SentenceTransformer model.
    It takes a list of sentences as input and returns a matrix of similarity scores.
    '''
    # Load the pre-trained model
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # Encode the sentences and calculate similarity scores
    embeddings = model.encode(sentences)
    similarity_scores = cosine_similarity(embeddings)

    return similarity_scores