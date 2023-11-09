from sentence_transformers import CrossEncoder
import numpy as np

# Function to predict the relationship between two sentences
# using the pre-trained model 'cross-encoder/nli-deberta-v3-small'
# from Hugging Face Transformers.
# The function takes two sentences as input and returns the predicted relationship.
# The possible relationships are 'contradiction', 'entailment', and 'neutral'.
def predict_relationship(sentence1: str, sentence2: str) -> str:
    # Load the pre-trained model
    model = CrossEncoder('cross-encoder/nli-deberta-v3-small')
    # Predict the scores for the sentence pair
    scores = model.predict([(sentence1, sentence2)])
    # Get the label with the highest score
    relationship = ['contradiction', 'entailment', 'neutral'][np.argmax(scores)]
    return relationship