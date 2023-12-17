# requirements_file --------------------

!pip install -U torch transformers

# function_import --------------------

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# function_code --------------------

def generate_jupiter_overview(text, question):
    """
    Generates an overview about how Jupiter became the largest planet in our solar system
    by answering a specific question using a pre-trained model.

    Args:
        text (str): The context text containing information about Jupiter.
        question (str): The question to be answered regarding Jupiter.

    Returns:
        str: The answer to the question based on the context text.
    """
    tokenizer = AutoTokenizer.from_pretrained('valhalla/longformer-base-4096-finetuned-squadv1')
    model = AutoModelForQuestionAnswering.from_pretrained('valhalla/longformer-base-4096-finetuned-squadv1')

    # Encode the inputs
    encoding = tokenizer(question, text, return_tensors='pt')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # Get answer from the model
    start_scores, end_scores = model(input_ids, attention_mask=attention_mask)
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    answer_tokens = all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores) + 1]
    answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
    return answer

# test_function_code --------------------

def test_generate_jupiter_overview():
    print("Testing started.")
    # Prepare the context and question
    text = "Jupiter is the largest planet in our Solar System..."
    question = "How did Jupiter become the largest planet?"

    # Test case 1: Correct answer
    print("Testing case [1/1] started.")
    answer = generate_jupiter_overview(text, question)
    assert answer != '', f"Test case [1/1] failed: Expected a non-empty answer, got an empty string."
    print("Testing case [1/1] passed.")
    print("Testing finished.")

# Run the test function
test_generate_jupiter_overview()