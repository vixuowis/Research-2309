# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import TapasTokenizer, TapasForQuestionAnswering

# function_code --------------------

def answer_financial_questions(transaction_data, queries):
    """
    Provides answers to questions about financial transactions within a given date range.

    Args:
        transaction_data (list of dict): A list of dictionaries with transaction data, each containing 'date', 'transaction', and 'monetary_value'.
        queries (list of str): Questions related to the number of transactions and their monetary value over specific date ranges.

    Returns:
        list: A list of answers to the given questions.

    Raises:
        ValueError: If transaction_data is not in the correct format or queries is empty.
    """
    tokenizer = TapasTokenizer.from_pretrained('google/tapas-small-finetuned-wikisql-supervised')
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-small-finetuned-wikisql-supervised')

    # Ensure the data is correctly formatted
    if not all(('date' in entry and 'transaction' in entry and 'monetary_value' in entry) for entry in transaction_data):
        raise ValueError('Transaction data must have date, transaction, and monetary_value fields')
    if not queries:
        raise ValueError('Queries list cannot be empty')

    inputs = tokenizer(table=transaction_data, queries=queries, return_tensors='pt')
    outputs = model(**inputs)
    predictions = tokenizer.convert_logits_to_predictions(inputs, outputs.logits)

    # Convert the predictions to a more readable format
    return [prediction for prediction in predictions[0]]

# test_function_code --------------------

def test_answer_financial_questions():
    print("Testing started.")
    # Fake financial data for testing
    transaction_data = [
        {'date': '2023-01-01', 'transaction': 'Deposit', 'monetary_value': 1000},
        {'date': '2023-02-01', 'transaction': 'Withdrawal', 'monetary_value': 200}
    ]

    # Testing cases
    queries = [
        "How many transactions occurred between 2023-01-01 and 2023-02-02?",
        "What is the total monetary value of transactions between 2023-01-01 and 2023-02-02?"
    ]
    expected_answers = [2, 1200] # Expected answers for the test queries

    for i, query in enumerate(queries):
        print(f"Testing case [{i+1}/{len(queries)}] started.")
        answer = answer_financial_questions(transaction_data, [query])[0]
        assert answer == expected_answers[i], f"Test case [{i+1}/{len(queries)}] failed: Expected {expected_answers[i]}, got {answer}"
        print(f"Testing case [{i+1}/{len(queries)}] finished.")
    print("Testing finished.")

# call_test_function_line --------------------

test_answer_financial_questions()