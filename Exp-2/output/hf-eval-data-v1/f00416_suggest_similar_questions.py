from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def suggest_similar_questions(user_question, available_questions):
    '''
    This function takes a user's question and a list of available questions, and returns the most similar question to the user's question.
    It uses the SentenceTransformer model from Hugging Face Transformers to generate sentence embeddings, and then calculates the cosine similarity between the user's question and the available questions.
    '''
    # Initialize the model
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
    
    # Generate the embedding for the user's question
    user_question_embedding = model.encode([user_question])
    
    # Generate the embeddings for the available questions
    available_questions_embeddings = model.encode(available_questions)
    
    # Calculate the cosine similarity between the user's question and the available questions
    similarities = cosine_similarity(user_question_embedding, available_questions_embeddings)
    
    # Find the index of the most similar question
    most_similar_index = np.argmax(similarities)
    
    # Return the most similar question
    return available_questions[most_similar_index]