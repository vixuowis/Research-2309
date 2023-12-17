# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def answer_inflation_question(context, question):
    # Initialize the question-answering pipeline with the BERT large cased model
    qa_pipeline = pipeline('question-answering', model='bert-large-cased-whole-word-masking-finetuned-squad')

    # Use the pipeline to find the answer to the question within the provided context
    result = qa_pipeline({'context': context, 'question': question})

    # Return the answer
    return result

# test_function_code --------------------

def test_answer_inflation_question():
    print("Testing answer_inflation_question function.")

    # Example of context related to inflation
    context = 'Inflation can occur when prices rise due to increases in production costs, such as raw materials and wages.'
    question = 'What causes inflation?'

    # Expected answer is related to the question about causes of inflation
    expected_answer = 'increases in production costs, such as raw materials and wages'

    # Call the function to test
    answer = answer_inflation_question(context, question)
    assert expected_answer in answer['answer'], f"Test failed: Expected \"{expected_answer}\" in \"{answer['answer']}\""

    print("All tests passed.")

# Run the test function
test_answer_inflation_question()