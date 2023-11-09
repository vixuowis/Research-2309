def test_generate_code():
    # Test the generate_code function with a sample description
    description = 'Write a Python function to calculate the factorial of a number.'
    expected_output = 'def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)'
    output = generate_code(description)
    # Do not compare the output strictly, as the generated code can vary
    assert 'factorial' in output
    assert 'n' in output
    assert '*' in output
    assert 'return' in output