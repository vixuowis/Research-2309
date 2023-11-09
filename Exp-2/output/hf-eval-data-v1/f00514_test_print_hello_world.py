def test_print_hello_world():
    """
    This function tests the print_hello_world function.
    It captures the output of the function and asserts that it matches the expected output.
    """
    import io
    import sys
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        print_hello_world()
    out = f.getvalue()

    assert out == 'Hello, World!\n', 'Test failed: Expected output does not match actual output.'

test_print_hello_world()