# requirements_file --------------------

!pip install -U transformers pandas

# function_import --------------------

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def answer_table_query(csv_file, query):
    """
    Answer a query based on the contents of a csv table using a pre-trained NLP model.

    :param csv_file: The path to the CSV file containing the table data.
    :param query: The natural language query related to the table.
    :return: The answer extracted by the model.
    """
    tokenizer = AutoTokenizer.from_pretrained('neulab/omnitab-large-1024shot')
    model = AutoModelForSeq2SeqLM.from_pretrained('neulab/omnitab-large-1024shot')
    table = pd.read_csv(csv_file)

    encoding = tokenizer(table=table, query=query, return_tensors='pt')
    outputs = model.generate(**encoding)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return answer

# test_function_code --------------------

def test_answer_table_query():
    print("Testing started.")
    # Assuming 'olympics.csv' contains historical Olympics data
    csv_file = 'olympics.csv'  # Replace with your actual CSV file path
    test_cases = [
        {'query': 'In which year did beijing host the Olympic Games?', 'expected_answer': '2008'},
        {'query': 'Which city hosted the 1896 Olympic Games?', 'expected_answer': 'Athens'}
    ]

    for i, test in enumerate(test_cases):
        print(f"Testing case [{i+1}/{len(test_cases)}] started.")
        answer = answer_table_query(csv_file, test['query'])
        assert answer == test['expected_answer'], f"Test case [{i+1}/{len(test_cases)}] failed: Expected {test['expected_answer']}, got {answer}"
        print(f"Testing case [{i+1}/{len(test_cases)}] finished successfully.")

    print("Testing finished.")

test_answer_table_query()