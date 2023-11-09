# function_import --------------------

from transformers import AutoModelForTableQuestionAnswering, AutoTokenizer, pipeline

# function_code --------------------

def get_survey_results(survey_table, query):
    """
    Analyze survey results and get the number of respondents who have given a specific answer for each question.

    Args:
        survey_table (dict): The survey table in appropriate format.
        query (str): The question to be asked to the model.

    Returns:
        dict: The result of the query.

    Raises:
        ValueError: If the survey_table or query is not in the correct format.
    """
    tapas_model = AutoModelForTableQuestionAnswering.from_pretrained('navteca/tapas-large-finetuned-wtq')
    tapas_tokenizer = AutoTokenizer.from_pretrained('navteca/tapas-large-finetuned-wtq')

    nlp = pipeline('table-question-answering', model=tapas_model, tokenizer=tapas_tokenizer)
    result = nlp({'table': survey_table, 'query': query})
    return result

# test_function_code --------------------

def test_get_survey_results():
    """
    Test the get_survey_results function.
    """
    survey_table = {'Repository': ['Transformers', 'Datasets', 'Tokenizers'], 'Stars': ['36542', '4512', '3934'], 'Contributors': ['651', '77', '34'], 'Programming language': ['Python', 'Python', 'Rust, Python and NodeJS']}
    query = 'How many stars does the transformers repository have?'
    result = get_survey_results(survey_table, query)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'answer' in result, 'The result should contain an answer.'

# call_test_function_code --------------------

test_get_survey_results()