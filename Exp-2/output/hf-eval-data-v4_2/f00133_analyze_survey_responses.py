# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelForTableQuestionAnswering, AutoTokenizer, pipeline

# function_code --------------------

def analyze_survey_responses(survey_table, query):
    """
    Analyze survey responses using a pretrained TAPAS model to answer questions.

    Args:
        survey_table (dict): The survey table data in a dictionary format where keys are column names.
        query (str): The query asking about specific survey responses.

    Returns:
        dict: The result returned by the TAPAS model pipeline.

    Raises:
        ValueError: If survey_table is not provided or not in the correct format.
        ValueError: If query is not provided or empty.
    """
    if not survey_table or not isinstance(survey_table, dict):
        raise ValueError("Invalid or no survey table provided.")
    if not query or not isinstance(query, str):
        raise ValueError("Query is not provided or is not a string.")

    # Load the TAPAS model and tokenizer
    tapas_model = AutoModelForTableQuestionAnswering.from_pretrained('navteca/tapas-large-finetuned-wtq')
    tapas_tokenizer = AutoTokenizer.from_pretrained('navteca/tapas-large-finetuned-wtq')

    # Create the pipeline
    nlp_pipeline = pipeline('table-question-answering', model=tapas_model, tokenizer=tapas_tokenizer)

    # Use the pipeline to process the survey table and answer the query
    result = nlp_pipeline({'table': survey_table, 'query': query})
    return result

# test_function_code --------------------

def test_analyze_survey_responses():
    print("Testing started.")
    # Mock survey data
    survey_table = {
        'Question': ['Q1', 'Q2', 'Q3'],
        'Option A': [20, 15, 25],
        'Option B': [30, 25, 5],
        'Option C': [50, 60, 70]
    }
    query = "How many respondents chose option A for question 1?"

    # Test case 1
    print("Testing case [1/1] started.")
    result = analyze_survey_responses(survey_table, query)
    assert result['answer'] == '20', f"Test case [1/1] failed: Expected 20 but got {result['answer']}"
    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_survey_responses()