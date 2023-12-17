# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# function_code --------------------

def find_most_relevant_passage(question, passages):
    """Find the most relevant passage to a given question from a list of passages.

    Args:
        question (str): The question to be answered.
        passages (list): A list of passages as potential answers.

    Returns:
        str: The passage most likely to contain the answer to the question.
    """
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2')
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2')
    
    features = tokenizer([question] * len(passages), passages, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        scores = model(**features).logits
    sorted_passages = [passages[idx] for idx in scores.argsort(descending=True)]
    return sorted_passages[0]

# test_function_code --------------------

def test_find_most_relevant_passage():
    print("Testing started.")
    question = "How many people live in Berlin?"
    passages = [
        "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.", 
        "New York City is famous for the Metropolitan Museum of Art."
    ]
    correct_answer = passages[0]

    print("Testing case [1/1] started.")
    assert find_most_relevant_passage(question, passages) == correct_answer, "Test case [1/1] failed: The function did not return the correct passage."
    print("Testing finished.")

# call_test_function_line --------------------

test_find_most_relevant_passage()