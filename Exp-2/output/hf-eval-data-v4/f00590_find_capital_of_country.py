# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def find_capital_of_country(country_name: str, context: str) -> str:
    """
    This function uses a pre-trained language model to answer the question about
    the capital of a given country based on a provided context.

    :param country_name: The name of the country to find the capital of.
    :param context: The context in which the answer can be found.
    :return: The name of the capital city.
    """
    nlp = pipeline('question-answering', model='deepset/roberta-large-squad2')
    question = f"What is the capital of {country_name}?"
    answer = nlp({'question': question, 'context': context})
    return answer['answer']

# test_function_code --------------------

def test_find_capital_of_country():
    print("Testing started.")
    # Test case 1: Given the correct context about Germany
    print("Testing case [1/3] started.")
    context = "Berlin is the capital of Germany."
    assert find_capital_of_country('Germany', context) == 'Berlin', f"Test case [1/3] failed: Expected 'Berlin'"

    # Test case 2: Given the correct context about France
    print("Testing case [2/3] started.")
    context = "Paris is the capital and most populous city of France."
    assert find_capital_of_country('France', context) == 'Paris', f"Test case [2/3] failed: Expected 'Paris'"

    # Test case 3: Given incorrect context
    print("Testing case [3/3] started.")
    context = "London is the capital of the United Kingdom."
    assert find_capital_of_country('Germany', context) != 'Berlin', f"Test case [3/3] failed: Context does not contain information about Germany's capital."
    print("Testing finished.")

test_find_capital_of_country()