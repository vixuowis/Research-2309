# requirements_file --------------------

!pip install -U 

# function_import --------------------



# function_code --------------------

def print_hello_world():
    """
    This function prints 'Hello, World!' to the console.
    """
    # Print the greeting to the console
    print('Hello, World!')

# test_function_code --------------------

def test_print_hello_world():
    import io
    import sys

    # Capture the output of the print_hello_world function
    captured_output = io.StringIO()
    sys.stdout = captured_output
    print_hello_world()  # Call the function
    sys.stdout = sys.__stdout__

    # Check the result
    assert captured_output.getvalue().strip() == 'Hello, World!', "Function did not print 'Hello, World!'"

    print('test_print_hello_world passed.')