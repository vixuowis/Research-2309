# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import TapasTokenizer, TapasForQuestionAnswering

# function_code --------------------

def get_tallest_animal(animal_table):
    model_name = 'google/tapas-mini-finetuned-sqa'
    tokenizer = TapasTokenizer.from_pretrained(model_name)
    model = TapasForQuestionAnswering.from_pretrained(model_name)

    inputs = tokenizer(table=animal_table, queries='What is the tallest animal?', return_tensors="pt")
    outputs = model(**inputs)

    # Find the table cell with the tallest animal
    answer_index = outputs.predicted_answer_coordinates[0][0]
    tallest_animal = animal_table[answer_index[0]][answer_index[1]]

    return tallest_animal

# test_function_code --------------------

def test_get_tallest_animal():
    print("Testing get_tallest_animal function.")
    animal_table = [
        ['Animal', 'Height'],
        ['Giraffe', '5.5m'],
        ['Elephant', '3.3m'],
        ['Lion', '1.2m']
    ]

    expected_output = 'Giraffe'
    output = get_tallest_animal(animal_table)

    assert output == expected_output, f"Test failed: Expected {expected_output}, got {output}"
    print("Test passed successfully.")