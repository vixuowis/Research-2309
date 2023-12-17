# requirements_file --------------------

!pip install -U transformers pandas

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

# function_code --------------------

def query_average_lifespan(animal_name):
    """
    Query the average lifespan of the given animal from a predefined data table using NLP table QA.

    Args:
        animal_name (str): The name of the animal to query.

    Returns:
        str: The average lifespan of the animal.

    Raises:
        ValueError: If the animal_name is not in the data table.
    """
    # Define the table data
    data = {
        'Animal': ['Tiger', 'Lion', 'Giraffe', 'Elephant'],
        'Habitat': ['Forest', 'Grassland', 'Savanna', 'Savanna'],
        'Average Lifespan': [10, 12, 25, 50]
    }
    table = pd.DataFrame.from_dict(data)
  
    # Ensure the animal is in the table
    if animal_name not in table['Animal'].values:
        raise ValueError("Animal not found in the table.")
  
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('neulab/omnitab-large-finetuned-wtq')
    model = AutoModelForSeq2SeqLM.from_pretrained('neulab/omnitab-large-finetuned-wtq')
  
    # Frame the query
    query = f"What is the average lifespan of a {animal_name.lower()}?"
  
    # Encode the table and query
    encoding = tokenizer(table=table, query=query, return_tensors='pt')
  
    # Generate and decode the response
    outputs = model.generate(**encoding)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
  
    return answer

# test_function_code --------------------

def test_query_average_lifespan():
    print("Testing started.")
    # Test cases
    test_cases = [
        ('Giraffe', '25'),
        ('Tiger', '10'),
        ('Elephant', '50')
    ]

    for i, (animal, expected_lifespan) in enumerate(test_cases, 1):
        print(f"Testing case [{i}/{len(test_cases)}] started.")
        result = query_average_lifespan(animal)
        assert result == expected_lifespan, f"Test case [{i}/{len(test_cases)}] failed: Expected {expected_lifespan}, got {result}"
  
    print("Testing finished.")

# call_test_function_line --------------------

test_query_average_lifespan()