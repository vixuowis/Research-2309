# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# function_code --------------------

def retrieve_relevant_passage(question, passages):
    # Load pre-trained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2')
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2')

    # Tokenize input question and passages
    features = tokenizer([question]*len(passages), passages, padding=True, truncation=True, return_tensors='pt')
    # Generate predictions
    with torch.no_grad():
        scores = model(**features).logits
    # Sort passages based on scores
    sorted_passages = [passages[idx] for idx in scores.argsort(descending=True).squeeze()]
    return sorted_passages[0]

# test_function_code --------------------

def test_retrieve_relevant_passage():
    print("Testing started.")
    question = "How many people live in Berlin?"
    passages = [
        "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
        "New York City is famous for the Metropolitan Museum of Art."
    ]

    # Testing retrieval of the most relevant passage
    print("Testing retrieval started.")
    best_passage = retrieve_relevant_passage(question, passages)
    assert best_passage == passages[0], f"The retrieved passage is incorrect: got {best_passage}"
    print("Testing retrieval successful.")

    print("Testing finished.")

# Run the test function
test_retrieve_relevant_passage()