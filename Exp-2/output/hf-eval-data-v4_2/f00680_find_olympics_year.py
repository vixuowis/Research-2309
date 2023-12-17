# requirements_file --------------------

!pip install -U transformers pandas

# function_import --------------------

from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

# function_code --------------------

def find_olympics_year(city_name):
    """
    Find the year when the Olympic Games were held in a specific city.

    Args:
        city_name (str): The name of the city to query the Olympic year for.

    Returns:
        str: The year the Olympic Games were held in the specified city.

    Raises:
        ValueError: If the city name is not found in the list.
    """
    # Load the TAPEX tokenizer and model
    tokenizer = TapexTokenizer.from_pretrained('microsoft/tapex-base')
    model = BartForConditionalGeneration.from_pretrained('microsoft/tapex-base')
    
    # Create a DataFrame with Olympic Game host cities and their corresponding years
    data = {
        'year': [1896, 1900, 1904, 2004, 2008, 2012],
        'city': ['athens', 'paris', 'st. louis', 'athens', 'beijing', 'london']
    }
    table = pd.DataFrame.from_dict(data)
    
    # Set the query
    query = f"select year where city = {city_name}"
    
    # Tokenize the table and query using the TAPEX tokenizer
    encoding = tokenizer(table=table, query=query, return_tensors='pt')
    
    # Generate an answer to the query using the model
    outputs = model.generate(**encoding)
    
    # Decode the generated answer
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    if not answer:
        raise ValueError(f"City name {city_name} not found in the list.")
    
    return answer[0]

# test_function_code --------------------

def test_find_olympics_year():
    print("Testing started.")

    # Test case 1: Beijing
    print("Testing case [1/1] started.")
    year = find_olympics_year('beijing')
    assert year == '2008', f"Test case [1/1] failed: Expected '2008', got {year}."
    print("Testing finished.")

# call_test_function_line --------------------

test_find_olympics_year()