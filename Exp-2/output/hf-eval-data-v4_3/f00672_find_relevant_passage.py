# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# function_code --------------------

def find_relevant_passage(question: str, candidate_passages: list) -> str:
    """
    Finds the most relevant passage for a given question from a list of candidate passages.

    Args:
        question (str): A string representing the question.
        candidate_passages (list): A list of strings, each representing a candidate passage.

    Returns:
        str: The passage most relevant to the provided question.

    Raises:
        ValueError: If `candidate_passages` is empty.
    """

    if not candidate_passages:
        raise ValueError("candidate_passages list is empty.")

    # Load the pre-trained model and tokenizer
    model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Prepare the features for the model
    features = tokenizer([question] * len(candidate_passages), candidate_passages,
                         padding=True, truncation=True, return_tensors='pt')

    # Evaluate the model and get the relevance scores
    model.eval()
    with torch.no_grad():
        scores = model(**features).logits

    # Find the index of the passage with the highest relevance score
    highest_score_index = torch.argmax(scores).item()

    return candidate_passages[highest_score_index]

# test_function_code --------------------

def test_find_relevant_passage():
    print("Testing started.")

    # Test data
    question = "How many people live in Berlin?"
    candidate_passages = [
        "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
        "New York City is famous for the Metropolitan Museum of Art.",
    ]

    # Test case 1: Correct retrieval of the most relevant passage
    print("Testing case [1/1] started.")
    relevant_passage = find_relevant_passage(question, candidate_passages)
    assert relevant_passage == candidate_passages[0], f"Test case [1/1] failed: Expected '{candidate_passages[0]}', got '{relevant_passage}'"
    
    print("Testing finished.")

# call_test_function_line --------------------

test_find_relevant_passage()