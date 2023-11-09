from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Function to get the best answer to a question using a pretrained model
# The function takes a question and a list of possible answers as input
# It uses a pretrained model to rank the answers based on their relevance to the question
# The function returns the answer with the highest relevance score

def get_best_answer(question, passages):
    # Load the pretrained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2')
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2')

    # Tokenize the question and the possible answers
    features = tokenizer([question]*len(passages), passages, padding=True, truncation=True, return_tensors='pt')

    # Use the model to get relevance scores for the answers
    with torch.no_grad():
        scores = model(**features).logits

    # Sort the answers based on their scores
    sorted_passages = [passages[idx] for idx in scores.argsort(descending=True)]

    # Return the answer with the highest score
    return sorted_passages[0]