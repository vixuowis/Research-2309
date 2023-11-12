# function_import --------------------

from transformers import pipeline

# function_code --------------------

def table_question_answering(questions_list, table_data):
    '''
    This function uses the TAPAS model to answer questions related to tabular data.

    Args:
        questions_list (list): A list of questions to be answered by the model.
        table_data (dict): The table data in dictionary format.

    Returns:
        dict: The answers to the questions.
    '''
    tapas_pipeline = pipeline('table-question-answering', model='Meena/table-question-answering-tapas')
    answers = tapas_pipeline(questions_list, table_data)
    return answers

# test_function_code --------------------

def test_table_question_answering():
    '''
    This function tests the table_question_answering function.
    '''
    table_data = {
        'Actors': ['Brad Pitt', 'Leonardo Di Caprio', 'Margot Robbie'],
        'Movies': ['Fight Club', 'Titanic', 'Wolf of Wall Street'],
        'Year': [1999, 1997, 2013]
    }
    questions_list = ['Who acted in Fight Club?', 'Which movie did Leonardo Di Caprio act in?', 'When was Wolf of Wall Street released?']
    answers = table_question_answering(questions_list, table_data)
    assert len(answers) == len(questions_list), 'The number of answers does not match the number of questions.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_table_question_answering()