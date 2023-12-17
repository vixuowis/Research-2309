# requirements_file --------------------

!pip install -U transformers>=4.0.0

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def find_korean_stock_info(table, query):
    # Initialize the table-question-answering pipeline using the specified model
    table_qa = pipeline('table-question-answering', model='dsba-lab/koreapas-finetuned-korwikitq')

    # Ask the question and return the answer
    answer = table_qa(table=table, query=query)
    return answer


# test_function_code --------------------

def test_find_korean_stock_info():
    print("Testing find_korean_stock_info function.")
    table = {'header': ['company', 'stock price', 'market cap'], 'rows': [['samsung', 50000, 100000], ['lg', 30000, 45000]]}
    query = 'Which company has a higher market cap?'
    expected_answer = {'answer': 'samsung'}

    # Perform the function call
    result = find_korean_stock_info(table, query)

    # Asserting the function's output
    assert result == expected_answer, f"Failed to identify the correct company: {result}"
    print("Test passed successfully.")

# Running the test
test_find_korean_stock_info()
