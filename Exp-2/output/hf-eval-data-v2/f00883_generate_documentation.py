# function_import --------------------

from transformers import AutoTokenizer, AutoModelWithLMHead, SummarizationPipeline

# function_code --------------------

def generate_documentation(tokenized_code):
    """
    Generate documentation for a given Python function.

    Args:
        tokenized_code (str): The tokenized Python function for which to generate documentation.

    Returns:
        str: The generated documentation for the given Python function.
    """
    pipeline = SummarizationPipeline(
        model=AutoModelWithLMHead.from_pretrained('SEBIS/code_trans_t5_base_code_documentation_generation_python'),
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
    expected_documentation = 'The function e takes a message and an optional exit code as arguments. It prints the message with the print_log function, using YELLOW and BOLD as parameters. If an exit code is provided, it calls sys.exit with the exit code.'
    assert generate_documentation(tokenized_code) == expected_documentation

# call_test_function_code --------------------

test_generate_documentation()