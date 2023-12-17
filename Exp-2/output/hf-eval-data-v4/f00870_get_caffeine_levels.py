# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelForTableQuestionAnswering, AutoTokenizer, pipeline

# function_code --------------------

def get_caffeine_levels(menu_table, query):
    # Load model & tokenizer
    tapas_model = AutoModelForTableQuestionAnswering.from_pretrained('navteca/tapas-large-finetuned-wtq')
    tapas_tokenizer = AutoTokenizer.from_pretrained('navteca/tapas-large-finetuned-wtq')

    # Initialize the pipeline for table-question-answering
    nlp = pipeline('table-question-answering', model=tapas_model, tokenizer=tapas_tokenizer)

    # Execute the query and get the result
    result = nlp({'table': menu_table, 'query': query})

    # Return the result
    return result

# test_function_code --------------------

def test_get_caffeine_levels():
    print("Testing get_caffeine_levels function.")
    menu_table = {
        'Coffee Type': ['Espresso', 'Cappuccino', 'Latte', 'Americano', 'Mocha'],
        'Size': ['Small', 'Medium', 'Large'],
        'Caffeine Content': ['95 mg', '120 mg', '145 mg', '165 mg', '185 mg']
    }
    query = 'What are the caffeine levels in each cup of coffee?'

    # Call the function with the table and query
    result = get_caffeine_levels(menu_table, query)

    # Check if the result is structured as expected
    assert 'answer' in result, "Test case failed: 'answer' field is missing in the result."
    print("Test case passed: Correct structure of the result.")

    # Further tests can be added to verify the correctness of the content

# Execute the test function
test_get_caffeine_levels()