# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# function_code --------------------

def find_most_relevant_passage(question: str, candidate_passages: list) -> str:
    """
    Find the most relevant passage for a given question from a list of candidate passages.

    :param question: The question in natural language.
    :param candidate_passages: A list of candidate passages.
    :return: The passage that is most relevant to the given question.
    """
    # Load the pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Tokenize the question and candidate passages
    features = tokenizer([question] * len(candidate_passages), candidate_passages, padding=True, truncation=True, return_tensors='pt')
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
    
    # Find the most relevant passage
    most_relevant_passage = candidate_passages[scores.argmax()]
    return most_relevant_passage

# test_function_code --------------------

def test_find_most_relevant_passage():
    print("Testing started.")
    question = "How many people live in Berlin?"
    candidate_passages = [
        "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
        "New York City is famous for the Metropolitan Museum of Art."
    ]

    # Expected the Berlin passage to be more relevant
    assert find_most_relevant_passage(question, candidate_passages) == candidate_passages[0], f"Test case failed: The passage about Berlin was expected to be most relevant"
    print("Testing finished.")

    # Run the test function
    test_find_most_relevant_passage()