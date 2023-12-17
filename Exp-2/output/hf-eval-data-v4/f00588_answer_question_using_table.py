# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import TapasTokenizer, TapasForQuestionAnswering

# function_code --------------------

def answer_question_using_table(table, question):
    tokenizer = TapasTokenizer.from_pretrained('lysandre/tapas-temporary-repo')
    model = TapasForQuestionAnswering.from_pretrained('lysandre/tapas-temporary-repo')
    inputs = tokenizer(table=table, queries=question, return_tensors='pt')
    outputs = model(**inputs)
    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(inputs, outputs.logits.detach(), outputs.logits_aggregation.detach())
    # Extract the answer from the coordinates
    answer = ''
    for coordinate in predicted_answer_coordinates[0]:
        answer += table[coordinate[0]][coordinate[1]]
        answer += ' '
    return answer

# test_function_code --------------------

def test_answer_question_using_table():
    print("Testing started.")
    # Suppose the `table` and `question` are defined as follows for the test
    table = {'city': ['Amsterdam', 'Berlin', 'Paris'], 'population': [821752, 3562166, 2140526]}
    question = "What is the population of Paris?"

    # Expected answer based on the table
    expected_answer = "2140526 "

    # Run the function
    answer = answer_question_using_table(table, question)

    # Test if the function returns the expected answer
    assert answer.strip() == expected_answer.strip(), f"Test failed: Expected '{{expected_answer}}', got '{{answer}}'"
    print("Testing succeeded.")

# Run the test function
test_answer_question_using_table()