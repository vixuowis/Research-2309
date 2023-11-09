def test_generate_code_summary():
    """
    This function tests the generate_code_summary function.
    It uses a sample code snippet and checks if the output summary is a string.
    """
    code_snippet = 'def greet(user): print(f\'Hello, {user}!\')'
    summary = generate_code_summary(code_snippet)
    assert isinstance(summary, str), 'The output summary should be a string.'