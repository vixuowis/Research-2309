# requirements_file --------------------

!pip install -U transformers pandas

# function_import --------------------

from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd
from datetime import datetime

# function_code --------------------

def answer_transaction_questions(transaction_data, date_range, questions):
    """
    Answer questions about financial transactions within a given date range using TAPAS model.
    
    :param transaction_data: a pandas DataFrame containing the financial transactions.
                             The DataFrame must have columns named 'date', 'transaction_id', 'monetary_value', etc.
    :param date_range: a tuple of two strings (start_date, end_date) specifying the date range for the transactions.
    :param questions: a list of strings containing questions to be answered related to the transaction data.
    :return: a list of answers corresponding to the questions asked.
    """
    # Initialize the tokenizer and model
    tokenizer = TapasTokenizer.from_pretrained('google/tapas-small-finetuned-wikisql-supervised')
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-small-finetuned-wikisql-supervised')
    
    # Filter the transaction data based on the given date range
    filtered_data = transaction_data[
        (transaction_data['date'] >= date_range[0]) &
        (transaction_data['date'] <= date_range[1])
    ]
    
    # Prepare the data and questions for TAPAS input
    inputs = tokenizer(table=filtered_data, queries=questions, return_tensors="pt", padding='max_length', truncation=True)
    
    # Get the model's output
    outputs = model(**inputs)
    
    # Convert the logits to actual predictions
    predicted_answers = tokenizer.convert_logits_to_predictions(inputs, outputs.logits)
    
    # Return only the predicted text answers
    answers = [answer[0] for answer in predicted_answers[0]]
    return answers

# test_function_code --------------------

def test_answer_transaction_questions():
    print("Testing started.")
    # Create sample transaction data
    sample_data = pd.DataFrame({
        'date': [datetime.strptime(date, '%Y-%m-%d').date() for date in ['2022-01-01', '2022-01-05', '2022-01-10']],
        'transaction_id': [101, 102, 103],
        'monetary_value': [100.00, 150.00, 200.00]
    })
    
    date_range = ('2022-01-01', '2022-01-10')
    questions = [
        "How many transactions occurred?",
        "What is the total monetary value?"
    ]

    # Test case 1: Check for the correct number of transactions
    print("Testing case [1/2] started.")
    answers = answer_transaction_questions(sample_data, date_range, [questions[0]])
    assert len(answers) == 1 and answers[0].isdigit() and int(answers[0]) == 3, f"Test case [1/2] failed: Expected 3 transactions, got {answers[0]}"

    # Test case 2: Check for the total monetary value
    print("Testing case [2/2] started.")
    answers = answer_transaction_questions(sample_data, date_range, [questions[1]])
    assert len(answers) == 1 and float(answers[0]) == 450.00, f"Test case [2/2] failed: Expected total value 450.00, got {answers[0]}"
    print("Testing finished.")

# Run the test function
test_answer_transaction_questions()