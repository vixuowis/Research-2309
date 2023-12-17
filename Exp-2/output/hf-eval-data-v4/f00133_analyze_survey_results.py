# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelForTableQuestionAnswering, AutoTokenizer, pipeline

# function_code --------------------

def analyze_survey_results(survey_table, query):
    # Load the TAPAS model fine-tuned on WikiTable Questions
    tapas_model = AutoModelForTableQuestionAnswering.from_pretrained('navteca/tapas-large-finetuned-wtq')
    # Load the corresponding TAPAS tokenizer
    tapas_tokenizer = AutoTokenizer.from_pretrained('navteca/tapas-large-finetuned-wtq')
    # Create a pipeline for table-question-answering using the model and tokenizer
    nlp = pipeline('table-question-answering', model=tapas_model, tokenizer=tapas_tokenizer)
    # Use the pipeline to perform the query over the provided survey table
    result = nlp({'table': survey_table, 'query': query})
    return result

# test_function_code --------------------

def test_analyze_survey_results():
    print("Testing started.")
    # Example survey table
    survey_table = {
        'Question': ['What is your favorite fruit?', 'How often do you exercise?'],
        'Option A': ['Apple', 'Every day'],
        'Option B': ['Banana', 'Twice a week'],
        'Option C': ['Orange', 'Rarely'],
        'Responses': [150, 75]
    }
    # Test case 1: Favorite fruit is Apple
    query_1 = "How many respondents chose Apple as their favorite fruit?"
    result_1 = analyze_survey_results(survey_table, query_1)
    assert result_1['coordinates'][0] == (0, 3) and result_1['answer'] == '150', "Test case 1 failed."

    # Test case 2: Exercise twice a week
    query_2 = "How often do respondents exercise twice a week?"
    result_2 = analyze_survey_results(survey_table, query_2)
    assert result_2['coordinates'][0] == (1, 3) and result_2['answer'] == '75', "Test case 2 failed."

    print("Testing finished.")