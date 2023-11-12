# function_import --------------------

from transformers import AutoModelForTableQuestionAnswering, AutoTokenizer, pipeline

# function_code --------------------

def get_caffeine_levels(table: dict, query: str) -> str:
    '''
    Get the caffeine levels in each cup of coffee from the provided table.

    Args:
        table (dict): A dictionary representing the coffee shop menu. It should have keys such as 'Coffee Type', 'Size', and 'Caffeine Content'.
        query (str): A string representing the question to be answered by the model.

    Returns:
        str: The answer to the query provided by the model.
    '''
    tapas_model = AutoModelForTableQuestionAnswering.from_pretrained('navteca/tapas-large-finetuned-wtq')
    tapas_tokenizer = AutoTokenizer.from_pretrained('navteca/tapas-large-finetuned-wtq')
    nlp = pipeline('table-question-answering', model=tapas_model, tokenizer=tapas_tokenizer)
    result = nlp({'table': table, 'query': query})
    return result

# test_function_code --------------------

def test_get_caffeine_levels():
    '''
    Test the get_caffeine_levels function.
    '''
    menu_table = {
        'Coffee Type': ['Espresso', 'Cappuccino', 'Latte', 'Americano', 'Mocha'],
        'Size': ['Small', 'Medium', 'Large'],
        'Caffeine Content': ['95 mg', '120 mg', '145 mg', '165 mg', '185 mg']
    }
    query = 'What are the caffeine levels in each cup of coffee?'
    result = get_caffeine_levels(menu_table, query)
    assert isinstance(result, str), 'The result should be a string.'
    assert 'mg' in result, 'The result should contain the unit of caffeine content.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_caffeine_levels()