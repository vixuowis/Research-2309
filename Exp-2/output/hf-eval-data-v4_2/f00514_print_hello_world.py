# requirements_file --------------------



# function_import --------------------



# function_code --------------------

def print_hello_world():
    """Prints 'Hello, World!' to the console.\n\n    Args: None\n    Returns: None\n    Raises: None\n    """
    print('Hello, World!')

# test_function_code --------------------

def test_print_hello_world():
    import io
    import sys

    print('Testing started.')
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Testing case 1: Check if 'Hello, World!' is printed
    print('Testing case [1/1] started.')
    print_hello_world()
    sys.stdout = sys.__stdout__
    assert 'Hello, World!' in captured_output.getvalue(), 'Test case [1/1] failed: Expected print output not found.'

    print('Testing finished.')

# call_test_function_line --------------------

test_print_hello_world()