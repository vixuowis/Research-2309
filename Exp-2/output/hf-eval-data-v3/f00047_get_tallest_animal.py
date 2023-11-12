# function_import --------------------

from transformers import TapasTokenizer, TapasForQuestionAnswering

# function_code --------------------

def get_tallest_animal(animal_table):
    """
    This function uses the 'google/tapas-mini-finetuned-sqa' model to perform the Table Question Answering task.
    It parses the provided table containing information about animals and their characteristics.
    It queries the model to retrieve the required information about the tallest animal in the table.

    Args:
        animal_table (list): A list of lists representing the table of animals and their characteristics.

    Returns:
        str: The name of the tallest animal in the table.

    Raises:
        ValueError: If the model fails to parse the table or answer the query.
    """
    model_name = 'google/tapas-mini-finetuned-sqa'
    tokenizer = TapasTokenizer.from_pretrained(model_name)
    model = TapasForQuestionAnswering.from_pretrained(model_name)
    inputs = tokenizer(table=animal_table, queries='What is the tallest animal?', return_tensors='pt')
    outputs = model(**inputs)
    answer_index = outputs.predicted_answer_coordinates[0][0]
    tallest_animal = animal_table[answer_index[0]][answer_index[1]]
    return tallest_animal

# test_function_code --------------------

def test_get_tallest_animal():
    """
    This function tests the get_tallest_animal function with a sample table.
    It asserts that the function returns the correct answer.
    """
    sample_table = [['Animal', 'Height'], ['Giraffe', '5.5m'], ['Elephant', '3.3m'], ['Lion', '1.2m']]
    assert get_tallest_animal(sample_table) == 'Giraffe'
    sample_table = [['Animal', 'Height'], ['Elephant', '3.3m'], ['Lion', '1.2m'], ['Giraffe', '5.5m']]
    assert get_tallest_animal(sample_table) == 'Giraffe'
    sample_table = [['Animal', 'Height'], ['Lion', '1.2m'], ['Giraffe', '5.5m'], ['Elephant', '3.3m']]
    assert get_tallest_animal(sample_table) == 'Giraffe'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_tallest_animal()