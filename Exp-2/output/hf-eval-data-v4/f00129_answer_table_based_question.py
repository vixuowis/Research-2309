# requirements_file --------------------

!pip install -U transformers pandas

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

# function_code --------------------

def answer_table_based_question(table_data, query):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('neulab/omnitab-large-finetuned-wtq')
    model = AutoModelForSeq2SeqLM.from_pretrained('neulab/omnitab-large-finetuned-wtq')

    # Create DataFrame from table data
    table = pd.DataFrame.from_dict(table_data)

    # Encode the table and question for the model
    encoding = tokenizer(table=table, query=query, return_tensors='pt')

    # Generate the answer
    outputs = model.generate(**encoding)

    # Decode the answer
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return answer[0]

# test_function_code --------------------

def test_answer_table_based_question():
    print("Testing answer_table_based_question function.")
    # Define a sample data table
    data = {
        'Animal': ['Tiger', 'Lion', 'Giraffe', 'Elephant'],
        'Habitat': ['Forest', 'Grassland', 'Savanna', 'Savanna'],
        'Average Lifespan': [10, 12, 25, 50],
    }
    # Define the question
    question = "What is the average lifespan of a giraffe?"

    # Expected answer
    expected_answer = "25"

    # Get the answer from the function
    answer = answer_table_based_question(data, question)

    # Test if the answer is correct
    assert answer == expected_answer, f"Failed: The answer '{{answer}}' is not equal to the expected answer '{{expected_answer}}'."
    print("All tests passed!")