# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import TapasForQuestionAnswering, TapasTokenizer

# function_code --------------------

def find_top_scorer(table_data, question):
    """
    Identifies the player who has scored the maximum goals in a given match.

    Args:
        table_data (str): The table data as a string in CSV format.
        question (str): The question to be answered by the model.

    Returns:
        str: The name of the player who scored the most goals.
    """
    # Load pre-trained TAPAS model
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-large-finetuned-sqa')
    tokenizer = TapasTokenizer.from_pretrained('google/tapas-large-finetuned-sqa')

    # Tokenize the data
    inputs = tokenizer(question, table_data, return_tensors="pt")
    outputs = model(**inputs)

    # Get the top scorer's name from the output
    answer_label = tokenizer.convert_ids_to_tokens(outputs.logits.argmax(axis=2)[0, 0])
    return answer_label

# test_function_code --------------------

def test_find_top_scorer():
    print("Testing find_top_scorer function.")
    # Test case: Find the top scorer from the provided table data
    table = "Player,Goals\nA,2\nB,3\nC,1"
    question = "What player scored the most goals?"
    expected_answer = "B"  # Expected top scorer
    answer = find_top_scorer(table, question)
    assert answer == expected_answer, f"Test failed: Expected {expected_answer}, got {answer}"
    print("Test passed: The top scorer identified correctly.")

# Run the test function
test_find_top_scorer()