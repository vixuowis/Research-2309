# requirements_file --------------------

!pip install -U transformers torch tensorflow

# function_import --------------------

from transformers import TapasForQuestionAnswering

# function_code --------------------

def get_revenue_for_year(question, table_data):
    # Load the pre-trained TAPAS model for WikiSQL supervised learning
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-base-finetuned-wikisql-supervised')
    
    # Pass the user's question and the table data to the model
    answer = model.predict(question, table_data)
    
    # Extract the answer from the output and return it
    return answer


# test_function_code --------------------

def test_get_revenue_for_year():
    print("Testing the get_revenue_for_year function.")

    # Define a sample question and table data
    question = "What was the revenue of the company in 2020?"
    table_data = [
      {"Year": "2018", "Revenue": "$20M"},
      {"Year": "2019", "Revenue": "$25M"},
      {"Year": "2020", "Revenue": "$30M"},
    ]

    # Expected answer for the sample question
    expected_answer = "$30M"

    # Test the function with the sample data
    answer = get_revenue_for_year(question, table_data)
    assert answer == expected_answer, f"Test failed: Expected {expected_answer}, got {answer}"

    print("Test passed successfully.")
