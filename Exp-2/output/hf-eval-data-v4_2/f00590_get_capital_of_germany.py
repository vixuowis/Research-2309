# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_capital_of_germany(question, context):
    """
    Finds the capital of Germany by using a pre-trained RoBERTa model for question answering.

    Args:
        question: A string representing the question to be answered.
        context: A string providing the context in which the question should be answered.

    Returns:
        A string representing the capital of Germany as extracted from the context.

    Raises:
        ValueError: If the context does not contain the answer to the question.
    """
    nlp = pipeline('question-answering', model='deepset/roberta-large-squad2')
    answer = nlp({'question': question, 'context': context})
    capital_of_germany = answer['answer']
    if not capital_of_germany:
        raise ValueError('The context does not contain the answer to the question.')
    return capital_of_germany


# test_function_code --------------------

def test_get_capital_of_germany():
    print("Testing started.")
    # Test case 1: Standard usage
    print("Testing case [1/3] started.")
    question = "What is the capital of Germany?"
    context = "Berlin is the capital and largest city of Germany."
    expected = "Berlin"
    result = get_capital_of_germany(question, context)
    assert result == expected, f"Test case [1/3] failed: Expected {{expected}}, got {{result}}"

    # Test case 2: Context without the answer
    print("Testing case [2/3] started.")
    context = "Germany is a country in Europe."
    try:
        get_capital_of_germany(question, context)
        assert False, "Test case [2/3] failed: ValueError not raised when context does not contain the answer."
    except ValueError:
        pass

    # Test case 3: Empty context
    print("Testing case [3/3] started.")
    context = ""
    try:
        get_capital_of_germany(question, context)
        assert False, "Test case [3/3] failed: ValueError not raised when context is empty."
    except ValueError:
        pass

    print("Testing finished.")


# call_test_function_line --------------------

test_get_capital_of_germany()