# function_import --------------------

from transformers import AutoModelForTableQuestionAnswering, AutoTokenizer, pipeline

# function_code --------------------

def get_caffeine_levels(table, query):
    """
    This function uses the TAPAS model from Hugging Face Transformers to answer questions related to a table.

    Args:
        table (dict): A dictionary representing the table data. The keys are the column names and the values are lists representing the column data.
        query (str): The question to be answered by the model.

    Returns:
        dict: The answer provided by the model.
    """
    tapas_model = AutoModelForTableQuestionAnswering.from_pretrained('navteca/tapas-large-finetuned-wtq')
    tapas_tokenizer = AutoTokenizer.from_pretrained('navteca/tapas-large-finetuned-wtq')
    nlp = pipeline('table-question-answering', model=tapas_model, tokenizer=tapas_tokenizer)
    result = nlp({'table': table, 'query': query})
    return result

# test_function_code --------------------

def test_get_caffeine_levels():
    """
    This function tests the get_caffeine_levels function by providing a sample table and a query.
    """
    table = {
        'Coffee Type': ['Espresso', 'Cappuccino', 'Latte', 'Americano', 'Mocha'],
        'Size': ['Small', 'Medium', 'Large'],
        'Caffeine Content': ['95 mg', '120 mg', '145 mg', '165 mg', '185 mg']
    }
    query = 'What are the caffeine levels in each cup of coffee?'
    result = get_caffeine_levels(table, query)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'answer' in result, 'The result dictionary should have an answer key.'

# call_test_function_code --------------------

test_get_caffeine_levels()