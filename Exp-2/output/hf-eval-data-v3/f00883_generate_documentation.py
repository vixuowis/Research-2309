# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, SummarizationPipeline

# function_code --------------------

def generate_documentation(tokenized_code):
    """
    Generate documentation for a given piece of Python code.

    Args:
        tokenized_code (str): The Python code to generate documentation for.

    Returns:
        str: The generated documentation.
    """
    pipeline = SummarizationPipeline(
        model=AutoModelForSeq2SeqLM.from_pretrained('SEBIS/code_trans_t5_base_code_documentation_generation_python'),
        tokenizer=AutoTokenizer.from_pretrained('SEBIS/code_trans_t5_base_code_documentation_generation_python', skip_special_tokens=True),
        device=0
    )
    return pipeline([tokenized_code])[0]['summary_text']

# test_function_code --------------------

def test_generate_documentation():
    """
    Test the generate_documentation function.
    """
    tokenized_code = 'def e(message, exit_code=None): print_log(message, YELLOW, BOLD) if exit_code is not None: sys.exit(exit_code)'
    expected_documentation = 'The function e takes two parameters: message and exit_code. It first prints a log with the message, YELLOW, and BOLD. If exit_code is not None, it then exits the system with exit_code.'
    assert generate_documentation(tokenized_code) == expected_documentation
    tokenized_code = 'def add(a, b): return a + b'
    expected_documentation = 'The function add takes two parameters: a and b. It returns the sum of a and b.'
    assert generate_documentation(tokenized_code) == expected_documentation
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_documentation()