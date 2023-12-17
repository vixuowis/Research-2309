# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import TapasTokenizer, TapasForQuestionAnswering

# function_code --------------------

def answer_table_question(model_name: str, question: str, table: dict) -> str:
    """
    Answers a question based on an inputted table using the TAPAS model from Hugging Face Transformers.

    Args:
        model_name (str): The pretrained model name or path.
        question (str): The natural language question to be answered.
        table (dict): The table containing the data, as a dictionary.

    Returns:
        str: The answer to the question based on the table.

    Raises:
        ValueError: If the answer cannot be determined from the table.

    """
    tokenizer = TapasTokenizer.from_pretrained(model_name)
    model = TapasForQuestionAnswering.from_pretrained(model_name)
    inputs = tokenizer(table=table, queries=question, return_tensors='pt')
    outputs = model(**inputs)
    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
        inputs,
        outputs.logits.detach(),
        outputs.logits_aggregation.detach()
    )

    # Extract the answer from the coordinates
    if predicted_answer_coordinates:
        answers = [table[i] for i in predicted_answer_coordinates[0]]
        return ' '.join(answers)
    else:
        raise ValueError('No answer could be determined from the table.')

# test_function_code --------------------

def test_answer_table_question():
    print("Testing started.")
    # Load a sample table and a question
    table = {'Position': ['1st', '2nd', '3rd'], 'Team': ['Team A', 'Team B', 'Team C'], 'Points': [90, 85, 75]}
    question = "Which team is in the 3rd position?"

    # Test case 1
    print("Testing case [1/1] started.")
    answer = answer_table_question('lysandre/tapas-temporary-repo', question, table)
    assert answer == 'Team C', f"Test case [1/1] failed: Expected 'Team C', but got {answer}"
    print("Testing finished.")

# call_test_function_line --------------------

test_answer_table_question()