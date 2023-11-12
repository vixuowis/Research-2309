# function_import --------------------

from transformers import AutoModelForTableQuestionAnswering, AutoTokenizer, pipeline

# function_code --------------------

def table_question_answering(survey_table: dict, query: str) -> dict:
    '''
    Analyze survey results and get the number of respondents who have given a specific answer for each question.

    Args:
        survey_table (dict): The survey table in appropriate format.
        query (str): The question to be answered.

    Returns:
        dict: The result of the query.
    '''
    tapas_model = AutoModelForTableQuestionAnswering.from_pretrained('navteca/tapas-large-finetuned-wtq')
    tapas_tokenizer = AutoTokenizer.from_pretrained('navteca/tapas-large-finetuned-wtq')
    nlp = pipeline('table-question-answering', model=tapas_model, tokenizer=tapas_tokenizer)
    result = nlp({'table': survey_table, 'query': query})
    return result

# test_function_code --------------------

def test_table_question_answering():
    '''
    Test the function table_question_answering.
    '''
    survey_table = {'Repository': ['Transformers', 'Datasets', 'Tokenizers'], 'Stars': ['36542', '4512', '3934'], 'Contributors': ['651', '77', '34'], 'Programming language': ['Python', 'Python', 'Rust, Python and NodeJS']}
    query = 'How many stars does the transformers repository have?'
    result = table_question_answering(survey_table, query)
    assert isinstance(result, dict)
    assert 'answer' in result
    assert 'coordinates' in result
    assert 'cells' in result
    assert 'score' in result
    return 'All Tests Passed'

# call_test_function_code --------------------

test_table_question_answering()