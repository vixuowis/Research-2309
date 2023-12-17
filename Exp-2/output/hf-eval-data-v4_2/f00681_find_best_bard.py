# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline, AutoTokenizer, AutoModelForTableQuestionAnswering

# function_code --------------------

def find_best_bard(table_data, question):
    """
    Finds the best bard with the highest magical ability from the given table data.

    Args:
        table_data (dict): The table containing bards data.
        question (str): The question to be asked.

    Returns:
        dict: The result returned by the table-question-answering pipeline.

    """
    tokenizer = AutoTokenizer.from_pretrained('google/tapas-mini-finetuned-wtq')
    model = AutoModelForTableQuestionAnswering.from_pretrained('google/tapas-mini-finetuned-wtq')
    nlp = pipeline('table-question-answering', model=model, tokenizer=tokenizer)
    return nlp({'table': table_data, 'query': question})

# test_function_code --------------------

def test_find_best_bard():
    print("Testing started.")
    # Sample table to test
    table_data = [
        {'Bard': 'Luna', 'Magical Ability': 'Moderate'},
        {'Bard': 'Cyrus', 'Magical Ability': 'High'},
        {'Bard': 'Aria', 'Magical Ability': 'Low'}
    ]

    question = "Which bard has the highest magical ability?"

    # Expected output is the bard with name 'Cyrus'
    expected = 'Cyrus'

    # Actual output using the function
    result = find_best_bard(table_data, question)
    assert result['answer'] == expected, f"Test case failed: Expected {expected}, got {result['answer']}"
    print("Testing finished.")

# call_test_function_line --------------------

test_find_best_bard()