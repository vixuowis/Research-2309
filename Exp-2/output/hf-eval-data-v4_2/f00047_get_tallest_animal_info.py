# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import TapasTokenizer, TapasForQuestionAnswering

# function_code --------------------

def get_tallest_animal_info(animal_table):
    """
    Retrieves information about the tallest animal from the given table.

    Args:
        animal_table (List[List[str]]): A two-dimensional list representation of the animal characteristics table.

    Returns:
        str: The name or characteristic of the tallest animal.

    """
    model_name = 'google/tapas-mini-finetuned-sqa'
    tokenizer = TapasTokenizer.from_pretrained(model_name)
    model = TapasForQuestionAnswering.from_pretrained(model_name)
    inputs = tokenizer(table=animal_table, queries='What is the tallest animal?', return_tensors="pt")
    outputs = model(**inputs)
    answer_index = outputs.predicted_answer_coordinates[0][0]
    tallest_animal = animal_table[answer_index[0]][answer_index[1]]
    return tallest_animal

# test_function_code --------------------

def test_get_tallest_animal_info():
    print("Testing started.")

    # Mock data representing a table of animals and their characteristics
    animal_table = [["Animal", "Height"],
                    ["Giraffe", "5.8"],
                    ["Elephant", "3.3"],
                    ["Lion", "1.2"]]

    # Expected result is "Giraffe" as it is the tallest
    expected_result = "Giraffe"

    print("Testing case [1/1] started.")
    tallest_animal = get_tallest_animal_info(animal_table)
    assert tallest_animal == expected_result, f"Test case [1/1] failed: Expected {expected_result}, got {tallest_animal}"
    print("Testing finished.")

# call_test_function_line --------------------

test_get_tallest_animal_info()