# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelForTableQuestionAnswering, AutoTokenizer, pipeline

# function_code --------------------

def get_coffee_caffeine_levels(menu_table):
    """Retrieve caffeine levels for all coffee types from the menu.

    Args:
        menu_table (dict): A dictionary representing the coffee menu.
                         It should have keys like 'Coffee Type' and
                         'Caffeine Content' with lists of values.

    Returns:
        dict: A dictionary with the query result.

    Raises:
        ValueError: If the provided menu_table is not valid.
    """
    # Load the TAPAS model pretrained on WikiTable Questions
    tapas_model = AutoModelForTableQuestionAnswering.from_pretrained('navteca/tapas-large-finetuned-wtq')
    # Load the tokenizer
    tapas_tokenizer = AutoTokenizer.from_pretrained('navteca/tapas-large-finetuned-wtq')
    # Create the NLP pipeline
    nlp = pipeline('table-question-answering', model=tapas_model, tokenizer=tapas_tokenizer)

    # Set the query for retrieving caffeine levels
    query = 'What are the caffeine levels in each cup of coffee?'
    # Get the result
    result = nlp({'table': menu_table, 'query': query})
    return result


# test_function_code --------------------

def test_get_coffee_caffeine_levels():
    print("Testing started.")
    # Sample coffee menu for testing
    sample_menu = {
        'Coffee Type': ['Espresso', 'Cappuccino', 'Latte', 'Americano', 'Mocha'],
        'Size': ['Small', 'Medium', 'Large'],
        'Caffeine Content': ['95 mg', '120 mg', '145 mg', '165 mg', '185 mg']
    }

    # Test case 1: Get caffeine levels for the normal menu
    print("Testing case [1/1] started.")
    result = get_coffee_caffeine_levels(sample_menu)
    assert type(result) is dict, f"Test case [1/1] failed: Expected dict, got {type(result)}"
    print("Testing finished.")


# call_test_function_line --------------------

test_get_coffee_caffeine_levels()