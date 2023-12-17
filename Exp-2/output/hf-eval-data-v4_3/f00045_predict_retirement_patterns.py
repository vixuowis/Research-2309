# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import TapasForQuestionAnswering, TapasTokenizer

# function_code --------------------

def predict_retirement_patterns(table_path, retirement_question, promotion_question):
    """
    Predicts retirement patterns and identifies top employees for potential promotions using TAPAS model.

    Args:
        table_path (str): The path to the CSV file containing employee data.
        retirement_question (str): A question to determine the average income and age of employees approaching retirement.
        promotion_question (str): A question to identify the top employees eligible for promotion.

    Returns:
        tuple: A tuple containing answers to the retirement and promotion questions.

    Raises:
        FileNotFoundError: An error occurs if the CSV file does not exist.
    """
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-large-finetuned-sqa')
    tokenizer = TapasTokenizer.from_pretrained('google/tapas-large-finetuned-sqa')

    try:
        with open(table_path, 'r') as file:
            table_data = file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f'The file at {table_path} does not exist.')

    inputs_retirement = tokenizer(table=table_data, queries=retirement_question, return_tensors="pt")
    inputs_promotion = tokenizer(table=table_data, queries=promotion_question, return_tensors="pt")

    retirement_output = model(**inputs_retirement)
    promotion_output = model(**inputs_promotion)

    retirement_answers = tokenizer.convert_logits_to_answers(table_data, **retirement_output)
    promotion_answers = tokenizer.convert_logits_to_answers(table_data, **promotion_output)

    return retirement_answers, promotion_answers

# test_function_code --------------------

def test_predict_retirement_patterns():
    print("Testing started.")
    table_path = "employee_data.csv"
    retirement_question = "What is the average annual income and age of employees who are close to retirement?"
    promotion_question = "Who are the top 5 employees with the highest performance ratings?"

    # Testing case [1/2] started
    print("Testing case [1/2] started.")
    try:
        retirement_answers, promotion_answers = predict_retirement_patterns(table_path, retirement_question, promotion_question)
        assert isinstance(retirement_answers, str) and isinstance(promotion_answers, str), f"Test case [1/2] failed: The answers must be string types."
    except FileNotFoundError as e:
        print(e)

    # Testing case [2/2]: Testing with non-existing file
    print("Testing case [2/2] started.")
    non_existing_table_path = "non_existing_file.csv"
    try:
        predict_retirement_patterns(non_existing_table_path, retirement_question, promotion_question)
        assert False, "Test case [2/2] failed: FileNotFoundError was expected."
    except FileNotFoundError:
        assert True

    print("Testing finished.")

# call_test_function_line --------------------

test_predict_retirement_patterns()