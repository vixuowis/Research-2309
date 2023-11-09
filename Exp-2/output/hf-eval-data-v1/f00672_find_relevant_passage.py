from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Function to find the most relevant passage given a question and several candidate passages
# Uses the Hugging Face Transformers library and a pre-trained model for sequence classification
# The model is trained on the MS Marco Passage Ranking task and can be used for Information Retrieval

def find_relevant_passage(question, candidate_passages):
    # Load the pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    # Load the pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    # Tokenize the given question and candidate passages
    features = tokenizer([question] * len(candidate_passages), candidate_passages, padding=True, truncation=True, return_tensors='pt')
    # Evaluate the model with the tokenized features
    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
    # Sort the passages in descending order based on the logits score
    sorted_passages = [x for _, x in sorted(zip(scores.detach().numpy(), candidate_passages), reverse=True)]
    # Return the most relevant passage
    return sorted_passages[0]