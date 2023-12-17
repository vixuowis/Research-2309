# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def answer_the_day_of_event(context, question):
    # This function uses a pre-trained model from Hugging Face Transformers to answer
    # the question about the day when the game was played given the context.

    # Create a pipeline for question answering using the specified model and tokenizer
    qa_pipeline = pipeline('question-answering', model='csarron/bert-base-uncased-squad-v1', tokenizer='csarron/bert-base-uncased-squad-v1')

    # Use the pipeline to predict the answer
    predictions = qa_pipeline({'context': context, 'question': question})

    # Return the predicted answer
    return predictions['answer']

# test_function_code --------------------

def test_answer_the_day_of_event():
    print("Testing started.")
    context = "The game was played on February 7, 2016 at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California."
    question = "What day was the game played on?"

    # Expected answer
    expected = 'February 7, 2016'

    # Test case
    print("Testing case [1/1] started.")
    answer = answer_the_day_of_event(context, question)
    assert answer == expected, f"Test case [1/1] failed: Expected '{expected}', but got '{answer}'"
    print("Testing finished.")

# Run the test function
test_answer_the_day_of_event()